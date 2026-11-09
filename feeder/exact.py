"""Exact grids: one object per layer per day that keeps every cell of every step at the source's own precision.

Each step is stored one of two ways, both lossless:
- ``ints``: the GRIB packed integers X (decoded value = ((X * 2**E) + R) * 10**-D, as the decoder computes it), cut
  into ``tile_size`` x ``tile_size`` blocks; each block is a 2-D delta (``dxy``, wrapping in the block's unsigned
  type) or left as is (``raw``), byte-shuffled and compressed as one zstd frame;
- ``grib``: the source bytes exactly as fetched (one GRIB2 message, or the served .gz file).
``store: auto`` keeps whichever is smaller per step; ``ints`` falls back to ``grib`` for a step whose integers do not
rebuild the decoded values bit for bit (for example a message with a bitmap).

Object layout (little-endian): 24-byte head = magic ``XGRID001``, u64 header length, u64 index length; the header
(canonical JSON: grid, tiling, and per step the mode, R / E / D / template / bits, the integer type, the first blob and
the blob count, and sha256 of the integers, of the decoded float64 values and of the source bytes); the index (u64
offset, u64 length per blob, relative to the data start); then the blobs. A step can be read with three range reads.

Exactness is proven twice: when a step is added (the integers rebuild the decoder's float64 values bit for bit) and
after the object is built (``verify`` reads every step back from the bytes and checks both sha256 values).
"""

import concurrent.futures as cf
import gzip
import hashlib
import json
import os
import struct
import tempfile
import threading

import numpy as np
import zstandard

from . import canonical_json
from .grib import ints_to_values, values_to_ints

MAGIC = b"XGRID001"
HEAD = struct.Struct("<8sQQ")
SCHEMA = "exact-grid/1"
DTYPES = {"u1": np.uint8, "u2": np.uint16, "u4": np.uint32}
FORMULA = "value = ((X * 2**E) + R) * 10**-D in float64; 2**E and 10**-D built by repeated multiplication/division"


class ExactError(Exception):
    pass


# ---------------------------------------------------------------- integer blocks

def dtype_for(xmax):
    return "u1" if xmax <= 0xFF else ("u2" if xmax <= 0xFFFF else "u4")


def shuffle(a):
    a = np.ascontiguousarray(a)
    if a.dtype.itemsize == 1:
        return a.tobytes()
    return a.view(np.uint8).reshape(-1, a.dtype.itemsize).T.tobytes()


def unshuffle(b, dtype, shape):
    dt = np.dtype(dtype)
    if dt.itemsize == 1:
        return np.frombuffer(b, dt).reshape(shape).copy()
    return np.ascontiguousarray(np.frombuffer(b, np.uint8).reshape(dt.itemsize, -1).T).view(dt).reshape(shape)


def predict(t, predictor):
    if predictor == "raw":
        return t
    if predictor != "dxy":
        raise ExactError(f"unknown predictor {predictor}")
    r = t.copy()
    r[:, 1:] = t[:, 1:] - t[:, :-1]
    r[1:] = r[1:] - r[:-1].copy()
    return r


def unpredict(r, predictor):
    if predictor == "raw":
        return r
    return np.cumsum(np.cumsum(r, axis=0, dtype=r.dtype), axis=1, dtype=r.dtype)


def tile_slices(shape, T):
    for y0 in range(0, shape[0], T):
        for x0 in range(0, shape[1], T):
            yield slice(y0, min(shape[0], y0 + T)), slice(x0, min(shape[1], x0 + T))


def ints_sha256(ints):
    return hashlib.sha256(np.ascontiguousarray(ints, dtype="<u4").tobytes()).hexdigest()


def values_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


_tls = threading.local()


def _compressor(level):
    c = getattr(_tls, "c", None)
    if c is None or _tls.level != level:
        _tls.c = c = zstandard.ZstdCompressor(level=level, write_checksum=True)
        _tls.level = level
    return c


def _decompressor():
    d = getattr(_tls, "d", None)
    if d is None:
        _tls.d = d = zstandard.ZstdDecompressor()
    return d


# ---------------------------------------------------------------- building

class ExactCube:
    def __init__(self, shape, n_steps, staging_dir, tile_size=128, predictor="dxy", level=19, store="auto",
                 workers=4):
        if store not in ("auto", "ints", "grib"):
            raise ExactError(f"unknown store mode {store}")
        self.shape = tuple(int(x) for x in shape)
        self.n_steps = int(n_steps)
        self.T = int(tile_size)
        self.predictor = predictor
        self.level = int(level)
        self.store = store
        self.workers = max(1, int(workers))
        self.slices = list(tile_slices(self.shape, self.T))
        self.steps = [{"i": i, "mode": "missing"} for i in range(self.n_steps)]
        fd, self._path = tempfile.mkstemp(suffix=".xg", dir=staging_dir)
        self._fh = os.fdopen(fd, "w+b")
        self._index = []
        self._off = 0
        self.grid_meta = None

    def close(self):
        try:
            self._fh.close()
            os.unlink(self._path)
        except OSError:
            pass

    def _append(self, blob):
        self._fh.write(blob)
        self._index.append((self._off, len(blob)))
        self._off += len(blob)
        return len(self._index) - 1

    def encode_ints(self, ints):
        """[zstd blobs], dtype name, for one step of integers (shape = the cube's)."""
        dt = dtype_for(int(ints.max()) if ints.size else 0)
        a = np.ascontiguousarray(ints, dtype=DTYPES[dt])

        def enc(sl):
            t = np.ascontiguousarray(a[sl])
            return _compressor(self.level).compress(shuffle(predict(t, self.predictor)))

        if self.workers > 1 and len(self.slices) > 1:
            with cf.ThreadPoolExecutor(max_workers=self.workers) as pool:
                blobs = list(pool.map(enc, self.slices))
        else:
            blobs = [enc(sl) for sl in self.slices]
        return blobs, dt

    def add_step(self, i, label, values, packing, source, source_encoding="none"):
        """``values``: the decoder's float64 field; ``packing``: its R / E / D / template / bits (or None);
        ``source``: the exact bytes fetched (one GRIB2 message, or the served .gz file)."""
        v = np.asarray(values, dtype=np.float64)
        if v.shape != self.shape:
            raise ExactError("step shape does not match the cube")
        rec = {"i": int(i), "valid_utc": label, "values_sha256": values_sha256(v),
               "source_sha256": hashlib.sha256(source).hexdigest(), "source_bytes": len(source),
               "source_encoding": source_encoding,
               "packing": None if packing is None else {k: packing[k] for k in ("template", "R", "E", "D", "nbits")}}
        ints = values_to_ints(v, packing)
        rec["int_exact"] = ints is not None
        if ints is not None:
            rec["ints_sha256"] = ints_sha256(ints)
        mode = "grib"
        blobs = None
        if ints is not None and self.store in ("auto", "ints"):
            blobs, dt = self.encode_ints(ints)
            ints_bytes = sum(len(b) for b in blobs) + 16 * len(blobs)
            rec["ints_bytes"] = ints_bytes
            if self.store == "ints" or ints_bytes <= len(source):
                mode = "ints"
                rec["dtype"] = dt
        if mode == "ints":
            rec["blob"] = self._append(blobs[0])
            for b in blobs[1:]:
                self._append(b)
            rec["nblobs"] = len(blobs)
            rec["stored_bytes"] = sum(len(b) for b in blobs)
        else:
            rec["blob"] = self._append(bytes(source))
            rec["nblobs"] = 1
            rec["stored_bytes"] = len(source)
        rec["mode"] = mode
        self.steps[int(i)] = rec
        return rec

    def add_missing(self, i, label):
        self.steps[int(i)] = {"i": int(i), "valid_utc": label, "mode": "missing"}

    def header(self, meta):
        h = dict(meta)
        h.update({"schema": SCHEMA, "shape": list(self.shape), "tile_size": self.T, "predictor": self.predictor,
                  "compression": {"codec": "zstd", "level": self.level, "byte_shuffle": True},
                  "tile_order": "row-major blocks of tile_size x tile_size from cell (0, 0); edge blocks are smaller",
                  "formula": FORMULA, "n_steps": self.n_steps, "steps": self.steps})
        return h

    def to_bytes(self, meta):
        hdr = canonical_json(self.header(meta))
        idx = np.array(self._index, dtype="<u8").reshape(-1, 2).tobytes() if self._index else b""
        self._fh.flush()
        self._fh.seek(0)
        body = self._fh.read()
        return HEAD.pack(MAGIC, len(hdr), len(idx)) + hdr + idx + body

    def summary(self):
        modes = [s["mode"] for s in self.steps]
        return {"steps_ints": modes.count("ints"), "steps_grib": modes.count("grib"),
                "steps_missing": modes.count("missing")}


# ---------------------------------------------------------------- reading

class ExactReader:
    def __init__(self, read_range):
        """``read_range(offset, length) -> bytes``."""
        self._read = read_range
        magic, hlen, ilen = HEAD.unpack(self._read(0, HEAD.size))
        if magic != MAGIC:
            raise ExactError("not an exact-grid object")
        self.header = json.loads(self._read(HEAD.size, hlen).decode("utf-8"))
        if self.header.get("schema") != SCHEMA:
            raise ExactError("unknown exact-grid schema")
        raw = self._read(HEAD.size + hlen, ilen)
        self.index = np.frombuffer(raw, dtype="<u8").reshape(-1, 2) if ilen else np.zeros((0, 2), "<u8")
        self.data_off = HEAD.size + hlen + ilen
        self.shape = tuple(self.header["shape"])
        self.slices = list(tile_slices(self.shape, int(self.header["tile_size"])))

    @classmethod
    def from_bytes(cls, data):
        mv = memoryview(data)
        return cls(lambda off, ln: bytes(mv[off:off + ln]))

    def blob(self, k):
        off, ln = (int(x) for x in self.index[k])
        return self._read(self.data_off + off, ln)

    def step(self, i):
        return self.header["steps"][int(i)]

    def ints(self, i):
        """Packed integers of step i (uint32, grid shape), or None if the step is stored as the source message."""
        s = self.step(i)
        if s["mode"] != "ints":
            return None
        dt = DTYPES[s["dtype"]]
        out = np.empty(self.shape, dtype=np.uint32)
        d = _decompressor()
        for k, sl in enumerate(self.slices):
            shp = (sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            r = unshuffle(d.decompress(self.blob(s["blob"] + k)), dt, shp)
            out[sl] = unpredict(r, self.header["predictor"])
        return out

    def source(self, i):
        """The stored source bytes of a ``grib`` step."""
        s = self.step(i)
        if s["mode"] != "grib":
            return None
        return self.blob(s["blob"])

    def values(self, i, decode=None, field=None):
        """Decoded float64 values of step i (``decode`` is needed for a ``grib`` step), or None if missing."""
        s = self.step(i)
        if s["mode"] == "ints":
            return ints_to_values(self.ints(i), s["packing"]).reshape(self.shape)
        if s["mode"] == "grib":
            msg = self.source(i)
            if s.get("source_encoding") == "gzip":
                msg = gzip.decompress(msg)
            return np.asarray(decode(msg, field or {"grib": None}, self.shape).values, dtype=np.float64)
        return None


def verify(data, decode=None, field=None):
    """Decode-back of a whole object: every step rebuilt from the bytes and compared bit for bit (sha256 of the
    integers and of the float64 values; ``grib`` steps: sha256 of the stored bytes, and of the values the decoder
    gives for them when ``decode`` is given). Returns (ok, {checked, failed: [step, ...]})."""
    r = ExactReader.from_bytes(data)
    failed, checked = [], 0
    for s in r.header["steps"]:
        i = s["i"]
        if s["mode"] == "missing":
            continue
        checked += 1
        try:
            if s["mode"] == "ints":
                x = r.ints(i)
                ok = ints_sha256(x) == s["ints_sha256"] and \
                    values_sha256(ints_to_values(x, s["packing"])) == s["values_sha256"]
            else:
                src = r.source(i)
                ok = hashlib.sha256(src).hexdigest() == s["source_sha256"]
                if ok and decode is not None:
                    ok = values_sha256(r.values(i, decode, field)) == s["values_sha256"]
        except Exception:  # anything unreadable is a failed step
            ok = False
        if not ok:
            failed.append(i)
    return not failed, {"checked": checked, "failed": failed}


def manifest_record(header):
    """Per-object record for the day manifest: the format, the formula, and per step the packing factors and
    checksums (enough to prove each step without opening the object)."""
    steps = []
    for s in header["steps"]:
        rec = {"valid_utc": s.get("valid_utc"), "mode": s["mode"]}
        if s["mode"] != "missing":
            p = s.get("packing") or {}
            rec.update({"template": p.get("template"), "R": p.get("R"), "E": p.get("E"), "D": p.get("D"),
                        "nbits": p.get("nbits"), "ints_sha256": s.get("ints_sha256"),
                        "values_sha256": s["values_sha256"], "source_sha256": s["source_sha256"]})
        steps.append(rec)
    return {"format": SCHEMA, "formula": FORMULA, "shape": header["shape"], "tile_size": header["tile_size"],
            "predictor": header["predictor"], "steps": steps}
