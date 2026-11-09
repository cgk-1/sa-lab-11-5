"""Sparse cubes for gridded fields: a small, fast-to-read subset of an exact grid.

Kept per (cell, step): every value at or above a threshold, plus every step at a regular lattice (every Nth row and
column) and at a list of point cells (a 3x3 halo around each station's nearest grid point, and the cell picked by
the minimum of squared longitude/latitude degree differences).

Values are the GRIB packed integers X (u32) with the step's R / E / D in the metadata, so
value = ((X * 2**E) + R) * 10**-D rebuilds the decoder's float64 bit for bit; 0xFFFFFFFF marks a missing value.
A step whose integers cannot be recovered exactly (no packing description, or a value the formula does not
reproduce) is quantised to 0.01 instead and counted in ``max_quant_err`` / ``exact_cells``; a cube built with
``exact=True`` then fails its decode check. Stored as Parquet (delta-encoded columns, zstd).
"""

import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .grib import ints_to_values, power

MPH_PER_MS = 3600.0 / 1609.344
MISSING = 0xFFFFFFFF
FLAG_THRESHOLD, FLAG_LATTICE, FLAG_POINT, FLAG_DEGREE = 1, 2, 4, 8
QUANTISED = {"template": None, "R": 0.0, "E": 0, "D": 2, "nbits": None, "quantised": True}


class SparseCube:
    def __init__(self, shape, n_steps, threshold_ms, lattice_step=10, point_cells=None, degree_cells=None,
                 exact=True):
        self.shape = tuple(shape)
        self.n_steps = int(n_steps)
        self.threshold_ms = float(threshold_ms)
        self.exact = bool(exact)
        ny, nx = self.shape
        lat_mask = np.zeros(self.shape, dtype=bool)
        if lattice_step:
            lat_mask[::lattice_step, ::lattice_step] = True
        self._fixed = lat_mask.ravel()
        self._fixed_flags = np.where(self._fixed, FLAG_LATTICE, 0).astype(np.uint8)

        def clean(cells):
            return sorted({int(c) for c in (cells or []) if c is not None and 0 <= int(c) < ny * nx})

        self.point_cells = clean(point_cells)
        self.degree_cells = clean(degree_cells)
        for cells, flag in ((self.point_cells, FLAG_POINT), (self.degree_cells, FLAG_DEGREE)):
            if cells:
                pc = np.array(cells, dtype=np.int64)
                self._fixed[pc] = True
                self._fixed_flags[pc] |= flag
        self._parts = []
        self.steps_present = []
        self.packing = {}
        self.max_quant_err = 0.0
        self.exact_cells = 0
        self.total_cells = 0

    def add_step(self, step, values, packing=None):
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.size != self.shape[0] * self.shape[1]:
            raise ValueError("step shape does not match the cube")
        finite = np.isfinite(v)
        with np.errstate(invalid="ignore"):
            above = finite & (v >= self.threshold_ms)
        keep = above | self._fixed
        cells = np.nonzero(keep)[0]
        vals = v[cells]
        fin = np.isfinite(vals)
        pk = None
        if packing and not packing.get("bitmap"):
            pk = {k: packing[k] for k in ("template", "R", "E", "D", "nbits")}
        if pk is not None:
            x, err = self._ints(vals[fin], pk)
            if err.size and np.any(err != 0):
                pk = None
        if pk is None:
            pk = dict(QUANTISED)
            x, err = self._ints(vals[fin], pk)
        q = np.full(cells.size, MISSING, dtype=np.uint32)
        q[fin] = x
        if err.size:
            self.max_quant_err = max(self.max_quant_err, float(err.max()))
            self.exact_cells += int(np.count_nonzero(err == 0))
            self.total_cells += int(err.size)
        flags = self._fixed_flags[cells] | np.where(above[cells], FLAG_THRESHOLD, 0).astype(np.uint8)
        self._parts.append((np.full(cells.size, step, dtype=np.uint16), cells.astype(np.uint32), q, flags))
        self.steps_present.append(int(step))
        self.packing[str(int(step))] = pk

    @staticmethod
    def _ints(vals, pk):
        s = power(2.0, int(pk["E"]))
        d = power(10.0, -int(pk["D"]))
        x = np.rint((vals / d - float(pk["R"])) / s)
        if x.size and (x.min() < 0 or x.max() > MISSING - 1):
            raise ValueError("value outside the packed-integer range")
        x = x.astype(np.uint32)
        back = ints_to_values(x, pk)
        err = np.abs(back - vals)
        err[back.view(np.int64) == vals.view(np.int64)] = 0.0
        return x, err

    @property
    def lossless(self):
        return self.exact_cells == self.total_cells

    def table(self):
        if self._parts:
            st, ce, va, fl = (np.concatenate(x) for x in zip(*self._parts))
        else:
            st = np.zeros(0, np.uint16)
            ce = np.zeros(0, np.uint32)
            va = np.zeros(0, np.uint32)
            fl = np.zeros(0, np.uint8)
        return pa.table({"step": pa.array(st, pa.uint16()), "cell": pa.array(ce, pa.uint32()),
                         "value": pa.array(va, pa.uint32()), "flags": pa.array(fl, pa.uint8())})

    def checks(self):
        return {"max_quant_err": self.max_quant_err, "exact_cells": self.exact_cells, "total_cells": self.total_cells}

    def to_parquet(self, meta):
        t = self.table()
        md = dict(meta)
        md.update({"shape": list(self.shape), "n_steps": self.n_steps, "missing": MISSING,
                   "value": "packed integer X; value = ((X * 2**E) + R) * 10**-D with the step's packing",
                   "packing": self.packing, "threshold_ms": self.threshold_ms, "steps_present": self.steps_present,
                   "n_point_cells": len(self.point_cells), "n_degree_cells": len(self.degree_cells),
                   "exact": self.exact, **self.checks(),
                   "flags": {"threshold": FLAG_THRESHOLD, "lattice": FLAG_LATTICE, "point": FLAG_POINT,
                             "degree": FLAG_DEGREE}})
        t = t.replace_schema_metadata({b"sparse_cube": json.dumps(md, sort_keys=True).encode("utf-8")})
        buf = io.BytesIO()
        # rows are sorted by step then cell, so delta encoding makes the cell column (and the values) small
        pq.write_table(t, buf, compression="zstd", compression_level=9, use_dictionary=False,
                       column_encoding={"step": "DELTA_BINARY_PACKED", "cell": "DELTA_BINARY_PACKED",
                                        "value": "DELTA_BINARY_PACKED"})
        return buf.getvalue(), t.num_rows


def read_parquet(data):
    t = pq.read_table(io.BytesIO(data))
    meta = json.loads(t.schema.metadata[b"sparse_cube"].decode("utf-8"))
    return t, meta


def decode_check(data, cube):
    """Reads the stored bytes back and compares every row with what the cube holds in memory; a cube that claims
    exactness also needs every kept value to be exact."""
    t, meta = read_parquet(data)
    ref = cube.table()
    if t.num_rows != ref.num_rows or tuple(meta["shape"]) != cube.shape or meta["packing"] != cube.packing:
        return False
    for col in ("step", "cell", "value", "flags"):
        if not np.array_equal(t.column(col).to_numpy(), ref.column(col).to_numpy()):
            return False
    if cube.exact and not cube.lossless:
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
    x = t.column("value").to_numpy()[sel]
    pk = meta["packing"].get(str(int(step)))
    if pk is None:
        return out.reshape(ny, nx)
    vals = ints_to_values(x, pk)
    vals[x == MISSING] = np.nan
    out[cells] = vals
    return out.reshape(ny, nx)
