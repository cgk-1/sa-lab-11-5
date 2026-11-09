"""Exact sparse cubes for gridded fields.

Kept per (cell, step): every value at or above a threshold, plus every step at a regular lattice (every Nth row and
column) and at a list of point cells. Values are stored as u16 in units of 0.01 (m/s for wind), i.e. at the GRIB
packing precision of the source fields; 65535 marks a missing value. Stored as Parquet (zstd).
"""

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

MPH_PER_MS = 3600.0 / 1609.344
SCALE = 100.0
MISSING = 65535
FLAG_THRESHOLD, FLAG_LATTICE, FLAG_POINT = 1, 2, 4


class SparseCube:
    def __init__(self, shape, n_steps, threshold_ms, lattice_step=10, point_cells=None):
        self.shape = tuple(shape)
        self.n_steps = int(n_steps)
        self.threshold_ms = float(threshold_ms)
        ny, nx = self.shape
        lat_mask = np.zeros(self.shape, dtype=bool)
        if lattice_step:
            lat_mask[::lattice_step, ::lattice_step] = True
        self._fixed = lat_mask.ravel()
        self._fixed_flags = np.where(self._fixed, FLAG_LATTICE, 0).astype(np.uint8)
        self.point_cells = sorted({int(c) for c in (point_cells or []) if c is not None and 0 <= c < ny * nx})
        if self.point_cells:
            pc = np.array(self.point_cells, dtype=np.int64)
            self._fixed[pc] = True
            self._fixed_flags[pc] |= FLAG_POINT
        self._parts = []
        self.steps_present = []
        self.max_quant_err = 0.0
        self.exact_cells = 0
        self.total_cells = 0

    def add_step(self, step, values):
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size != self.shape[0] * self.shape[1]:
            raise ValueError("step shape does not match the cube")
        finite = np.isfinite(v)
        above = finite & (v >= self.threshold_ms)
        keep = above | self._fixed
        cells = np.nonzero(keep)[0]
        vals = v[cells]
        fin = np.isfinite(vals)
        if np.any(fin & ((vals < 0) | (vals * SCALE > MISSING - 1))):
            raise ValueError("value outside the u16 range")
        q = np.full(cells.size, MISSING, dtype=np.uint16)
        q[fin] = np.rint(vals[fin] * SCALE).astype(np.uint16)
        err = np.abs(q[fin] / SCALE - vals[fin])
        if err.size:
            self.max_quant_err = max(self.max_quant_err, float(err.max()))
            self.exact_cells += int(np.count_nonzero(err < 1e-6))
            self.total_cells += int(err.size)
        flags = self._fixed_flags[cells] | np.where(above[cells], FLAG_THRESHOLD, 0).astype(np.uint8)
        self._parts.append((np.full(cells.size, step, dtype=np.uint16), cells.astype(np.uint32), q, flags))
        self.steps_present.append(int(step))

    def table(self):
        if self._parts:
            st, ce, va, fl = (np.concatenate(x) for x in zip(*self._parts))
        else:
            st = np.zeros(0, np.uint16)
            ce = np.zeros(0, np.uint32)
            va = np.zeros(0, np.uint16)
            fl = np.zeros(0, np.uint8)
        return pa.table({"step": pa.array(st, pa.uint16()), "cell": pa.array(ce, pa.uint32()),
                         "value": pa.array(va, pa.uint16()), "flags": pa.array(fl, pa.uint8())})

    def to_parquet(self, meta):
        t = self.table()
        md = dict(meta)
        md.update({"shape": list(self.shape), "n_steps": self.n_steps, "scale": SCALE, "missing": MISSING,
                   "threshold_ms": self.threshold_ms, "steps_present": self.steps_present,
                   "n_point_cells": len(self.point_cells),
                   "flags": {"threshold": FLAG_THRESHOLD, "lattice": FLAG_LATTICE, "point": FLAG_POINT}})
        t = t.replace_schema_metadata({b"sparse_cube": json.dumps(md, sort_keys=True).encode("utf-8")})
        buf = io.BytesIO()
        pq.write_table(t, buf, compression="zstd", compression_level=9, use_dictionary=False)
        return buf.getvalue(), t.num_rows


def read_parquet(data):
    t = pq.read_table(io.BytesIO(data))
    meta = json.loads(t.schema.metadata[b"sparse_cube"].decode("utf-8"))
    return t, meta


def decode_check(data, cube):
    """Reads the stored bytes back and compares every row with what the cube holds in memory."""
    t, meta = read_parquet(data)
    ref = cube.table()
    if t.num_rows != ref.num_rows or tuple(meta["shape"]) != cube.shape:
        return False
    for col in ("step", "cell", "value", "flags"):
        if not np.array_equal(t.column(col).to_numpy(), ref.column(col).to_numpy()):
            return False
    return True


def dense_step(data, step):
    """Re-creates one step as a dense float array (NaN where nothing is kept)."""
    t, meta = read_parquet(data)
    ny, nx = meta["shape"]
    out = np.full(ny * nx, np.nan)
    st = t.column("step").to_numpy()
    sel = st == step
    cells = t.column("cell").to_numpy()[sel]
    vals = t.column("value").to_numpy()[sel].astype(np.float64)
    vals[vals == MISSING] = np.nan
    out[cells] = vals / SCALE
    return out.reshape(ny, nx)
