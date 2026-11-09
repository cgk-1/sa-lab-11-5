"""Exact grids: one object per layer per day that keeps every cell of every step at the source's own precision.

Each step is stored one of two ways, both lossless:
- ``ints``: the GRIB packed integers X (decoded value = ((X * 2**E) + R) * 10**-D, as the decoder computes it), cut
  into ``tile_size`` x ``tile_size`` blocks; each block is predicted, its residuals are zigzag-mapped, byte-shuffled
  and compressed as one zstd frame (format 2, below);
- ``grib``: the source bytes exactly as fetched (one GRIB2 message, or the served .gz file).
``store: auto`` keeps whichever is smaller per step; ``ints`` falls back to ``grib`` for a step whose integers do not
rebuild the decoded values bit for bit (for example a message with a bitmap).

Format 2 (magic ``XGRID002``, schema ``exact-grid/2``; written by this version):
- each block is predicted on its own (no block reads a neighbour), with the cells above and left of the block taken
  as 0: ``dxy`` predicts a + b - c, ``med`` the LOCO-I median edge detector (min(a, b) if c >= max(a, b), max(a, b)
  if c <= min(a, b), else a + b - c), ``raw`` predicts 0 (a = left, b = above, c = above-left). So the first cell
  is predicted 0, the rest of the first row by its left cell and the rest of the first column by the cell above;
- the residual r = (X - prediction) mod 2**w is taken in [-2**(w-1), 2**(w-1)) and stored as 2r (r >= 0) or
  -2r - 1 (r < 0) in w bits, w = 8 x the step's dtype size: small negative residuals become small numbers;
- with ``palette`` on, a step with at most 256 distinct integers stores u1 indices into its sorted table of those
  integers; the table is kept in the step's header record (``palette``: dtype, count, base64 of the little-endian
  array), so an object decodes on its own.
Format 1 (magic ``XGRID001``, schema ``exact-grid/1``; read only): ``dxy`` as a 2-D delta wrapping in the block's
unsigned type, or ``raw``; byte-shuffled; one zstd frame per block.

Object layout (little-endian): 24-byte head = magic, u64 header length, u64 index length; the header (canonical
JSON: grid, tiling, and per step the mode, R / E / D / template / bits, the integer type, the first blob and the blob
count, and sha256 of the integers, of the decoded float64 values and of the source bytes); the index (u64 offset,
u64 length per blob, relative to the data start); then the blobs. A step can be read with three range reads, and
one block of it with three range reads of its own.

Exactness is proven twice: when a step is added (the integers rebuild the decoder's float64 values bit for bit) and
after the object is built (``verify`` reads every step back from the bytes and checks both sha256 values).
"""

import base64
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

MAGIC_V1 = b"XGRID001"
MAGIC = b"XGRID002"
MAGICS = (MAGIC_V1, MAGIC)
HEAD = struct.Struct("<8sQQ")
SCHEMA_V1 = "exact-grid/1"
SCHEMA = "exact-grid/2"
SCHEMA_OF = {MAGIC_V1: SCHEMA_V1, MAGIC: SCHEMA}
PREDICTORS = ("dxy", "med", "raw")
PALETTE_MAX = 256
DTYPES = {"u1": np.uint8, "u2": np.uint16, "u4": np.uint32}
FORMULA = "value = ((X * 2**E) + R) * 10**-D in float64; 2**E and 10**-D built by repeated multiplication/division"
PREDICTION = ("per block, cells above and left of the block = 0; a = left, b = above, c = above-left; dxy: a + b - c; "
              "med: min(a, b) if c >= max(a, b), max(a, b) if c <= min(a, b), else a + b - c; raw: 0")
RESIDUAL = ("r = (X - prediction) mod 2**w in [-2**(w-1), 2**(w-1)), stored as 2r if r >= 0 else -2r - 1 in w bits; "
            "w = 8 x dtype size; palette steps store indices into the step's sorted table")
_DECODE_BATCH = 256  # blocks inverted together by the med decoder (bounds memory: ~17 MB of int32 per batch)


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


# ---- format 1 (read only)

def predict_v1(t, predictor):
    if predictor == "raw":
        return t
    if predictor != "dxy":
        raise ExactError(f"unknown predictor {predictor}")
    r = t.copy()
    r[:, 1:] = t[:, 1:] - t[:, :-1]
    r[1:] = r[1:] - r[:-1].copy()
    return r


def unpredict_v1(r, predictor):
    if predictor == "raw":
        return r
    return np.cumsum(np.cumsum(r, axis=0, dtype=r.dtype), axis=1, dtype=r.dtype)


# ---- format 2

def _check_predictor(predictor):
    if predictor not in PREDICTORS:
        raise ExactError(f"unknown predictor {predictor}")


def prediction(x, predictor):
    """Prediction of every cell of block ``x`` (int64, values in [0, 2**w)) from its original neighbours."""
    if predictor == "raw":
        return np.zeros_like(x)
    p = np.zeros((x.shape[0] + 1, x.shape[1] + 1), np.int64)
    p[1:, 1:] = x
    a, b, c = p[1:, :-1], p[:-1, 1:], p[:-1, :-1]
    if predictor == "dxy":
        return a + b - c
    mx, mn = np.maximum(a, b), np.minimum(a, b)
    return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


def zigzag(r, dtype):
    """Signed residuals (int64, any value; taken mod 2**w) -> w-bit zigzag codes."""
    bits = 8 * np.dtype(dtype).itemsize
    m = (1 << bits) - 1
    r = np.asarray(r, np.int64) & m
    r = np.where(r >= (1 << (bits - 1)), r - (1 << bits), r)
    return np.where(r < 0, -2 * r - 1, 2 * r).astype(dtype)


def unzigzag(z, work=np.int64):
    v = np.asarray(z).astype(work)  # in place from here: fewer temporaries on big stacks
    t = v & 1
    np.negative(t, out=t)
    v >>= 1
    v ^= t
    return v


def predict(t, predictor):
    """Zigzag residual codes of block ``t`` (unsigned; the codes keep its dtype)."""
    _check_predictor(predictor)
    x = np.asarray(t).astype(np.int64)
    return zigzag(x - prediction(x, predictor), t.dtype)


_DIAG = {}


def _diagonals(shape):
    """Flat indices (padded block, and residual block) of each anti-diagonal of a block of ``shape``: the cells of
    one anti-diagonal depend only on earlier ones, so each is rebuilt in one vector step."""
    if shape not in _DIAG:
        ny, nx = shape
        w = nx + 1
        out = []
        for d in range(ny + nx - 1):
            i = np.arange(max(0, d - nx + 1), min(ny - 1, d) + 1)
            j = d - i
            pos = (i + 1) * w + (j + 1)
            out.append((pos, i * nx + j, pos - 1, pos - w, pos - w - 1))
        _DIAG[shape] = out
    return _DIAG[shape]


def _unmed(r, m, work):
    """med inverse for blocks stacked on the LAST axis: r (ny, nx, n) residuals -> (ny, nx, n) values."""
    ny, nx, n = r.shape
    rr = r.reshape(ny * nx, n)
    x = np.zeros(((ny + 1) * (nx + 1), n), work)
    for pos, ri, ia, ib, ic in _diagonals((ny, nx)):
        a, b = x[ia], x[ib]
        g = a + b - x[ic]
        np.maximum(g, np.minimum(a, b), out=g)  # med(a, b, c) = a + b - c clipped to [min(a, b), max(a, b)]
        np.minimum(g, np.maximum(a, b), out=g)
        g += rr[ri]
        g &= m
        x[pos] = g
    return x.reshape(ny + 1, nx + 1, n)[1:, 1:]


def unpredict(z, predictor, dtype=None, blocks_last=False):
    """Inverse of ``predict`` for one block (2-D) or a stack of blocks of one shape (3-D: blocks first, or last with
    ``blocks_last``; the result has the same layout). Returns the values as int32 (w <= 16) or int64: the arithmetic
    is mod 2**w, which divides the working type's own wrap-around, so a sum that wraps there is still right."""
    _check_predictor(predictor)
    dt = np.dtype(dtype or z.dtype)
    work = np.int32 if dt.itemsize <= 2 else np.int64
    m = work((1 << (8 * dt.itemsize)) - 1)
    r = unzigzag(z, work)
    if r.ndim == 2:
        r, one = r[..., None], True
    else:
        r, one = (r if blocks_last else np.moveaxis(r, 0, -1)), False
    if predictor == "raw":
        x = r & m
    elif predictor == "dxy":
        x = np.cumsum(np.cumsum(r, axis=0, dtype=work), axis=1, dtype=work)
        x &= m
    else:
        r = np.ascontiguousarray(r)
        x = np.empty(r.shape, work)
        for s in range(0, r.shape[-1], _DECODE_BATCH):
            x[..., s:s + _DECODE_BATCH] = _unmed(np.ascontiguousarray(r[..., s:s + _DECODE_BATCH]), m, work)
    if one:
        return x[..., 0]
    return x if blocks_last else np.moveaxis(x, -1, 0)


def palette_of(ints, palette):
    """(sorted table, u1 indices) when ``palette`` is "auto" and the step has at most 256 distinct integers, else
    (None, None)."""
    if palette != "auto" or ints.size == 0:
        return None, None
    table = np.unique(ints)
    if table.size > PALETTE_MAX:
        return None, None
    return table, np.searchsorted(table, ints).astype(np.uint8)


def palette_record(table):
    dt = dtype_for(int(table[-1]))
    return {"dtype": dt, "count": int(table.size),
            "b64": base64.b64encode(np.ascontiguousarray(table, dtype="<" + np.dtype(DTYPES[dt]).str[1:])
                                    .tobytes()).decode("ascii")}


def palette_table(rec):
    t = np.frombuffer(base64.b64decode(rec["b64"]), dtype="<" + np.dtype(DTYPES[rec["dtype"]]).str[1:])
    if t.size != int(rec["count"]):
        raise ExactError("palette length does not match its count")
    return t.astype(np.uint32)


def tile_slices(shape, T):
    for y0 in range(0, shape[0], T):
        for x0 in range(0, shape[1], T):
            yield slice(y0, min(shape[0], y0 + T)), slice(x0, min(shape[1], x0 + T))


def ints_sha256(ints):
    # hashes the array's buffer in place (the same bytes .tobytes() would copy out)
    return hashlib.sha256(np.ascontiguousarray(ints, dtype="<u4").reshape(-1)).hexdigest()


def values_sha256(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").reshape(-1)).hexdigest()


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


# ---------------------------------------------------------------- one step (pure: runs in any process)

def encode_ints(ints, slices, predictor, level, workers=1, palette="off"):
    """[zstd blobs], dtype name, palette record (or None): each ``slices`` block of ``ints`` (or of its palette
    indices) predicted, zigzag-mapped, byte-shuffled and compressed as one zstd frame. The blobs depend only on the
    integers and (predictor, level, palette), never on ``workers``."""
    _check_predictor(predictor)
    table, idx = palette_of(ints, palette)
    if table is not None:
        dt, a, pal = "u1", idx, palette_record(table)
    else:
        dt = dtype_for(int(ints.max()) if ints.size else 0)
        a, pal = np.ascontiguousarray(ints, dtype=DTYPES[dt]), None

    def enc(sl):
        t = np.ascontiguousarray(a[sl])
        return _compressor(level).compress(shuffle(predict(t, predictor)))

    if workers > 1 and len(slices) > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as pool:
            blobs = list(pool.map(enc, slices))
    else:
        blobs = [enc(sl) for sl in slices]
    return blobs, dt, pal


def prepare_step(values, packing, source, source_encoding, shape, tile_size, predictor, level, store, workers=1,
                 slices=None, palette="off"):
    """(step record without its position, [blobs to append]) for one step of a cube with these parameters.

    ``ints`` mode when the packed integers rebuild ``values`` bit for bit and (``store`` auto) their blobs (plus the
    palette table) are no larger than the source; otherwise the source bytes as fetched."""
    shape = tuple(int(x) for x in shape)
    v = np.asarray(values, dtype=np.float64)
    if v.shape != shape:
        raise ExactError("step shape does not match the cube")
    rec = {"values_sha256": values_sha256(v),
           "source_sha256": hashlib.sha256(source).hexdigest(), "source_bytes": len(source),
           "source_encoding": source_encoding,
           "packing": None if packing is None else {k: packing[k] for k in ("template", "R", "E", "D", "nbits")}}
    ints = values_to_ints(v, packing)
    rec["int_exact"] = ints is not None
    if ints is not None:
        rec["ints_sha256"] = ints_sha256(ints)
    if ints is not None and store in ("auto", "ints"):
        blobs, dt, pal = encode_ints(ints, slices if slices is not None else list(tile_slices(shape, int(tile_size))),
                                     predictor, int(level), workers, palette)
        ints_bytes = sum(len(b) for b in blobs) + 16 * len(blobs)
        if pal is not None:
            ints_bytes += pal["count"] * np.dtype(DTYPES[pal["dtype"]]).itemsize
        rec["ints_bytes"] = ints_bytes
        if store == "ints" or ints_bytes <= len(source):
            rec["dtype"] = dt
            rec["mode"] = "ints"
            if pal is not None:
                rec["palette"] = pal
            return rec, blobs
    rec["mode"] = "grib"
    return rec, [bytes(source)]


# ---------------------------------------------------------------- building

class ExactCube:
    def __init__(self, shape, n_steps, staging_dir, tile_size=128, predictor="dxy", level=19, store="auto",
                 workers=4, palette="off"):
        if store not in ("auto", "ints", "grib"):
            raise ExactError(f"unknown store mode {store}")
        _check_predictor(predictor)
        if palette not in ("auto", "off"):
            raise ExactError(f"unknown palette mode {palette}")
        self.shape = tuple(int(x) for x in shape)
        self.n_steps = int(n_steps)
        self.T = int(tile_size)
        self.predictor = predictor
        self.level = int(level)
        self.store = store
        self.palette = palette
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
        """[zstd blobs], dtype name, palette record, for one step of integers (shape = the cube's)."""
        return encode_ints(ints, self.slices, self.predictor, self.level, self.workers, self.palette)

    def params(self):
        """What a worker process needs to prepare a step for this cube (see ``prepare_step``)."""
        return {"shape": self.shape, "tile_size": self.T, "predictor": self.predictor, "level": self.level,
                "store": self.store, "palette": self.palette}

    def add_step(self, i, label, values, packing, source, source_encoding="none"):
        """``values``: the decoder's float64 field; ``packing``: its R / E / D / template / bits (or None);
        ``source``: the exact bytes fetched (one GRIB2 message, or the served .gz file)."""
        prepared = prepare_step(values, packing, source, source_encoding, workers=self.workers, slices=self.slices,
                                **self.params())
        return self.commit_step(i, label, prepared)

    def commit_step(self, i, label, prepared):
        """Appends a step made by ``prepare_step`` (here or in a worker process) as step ``i``. Steps must be
        committed in the order the serial loop adds them, so the blob numbering is the same."""
        rec, blobs = prepared
        rec = dict(rec)
        rec["i"] = int(i)
        rec["valid_utc"] = label
        rec["blob"] = self._append(blobs[0])
        for b in blobs[1:]:
            self._append(b)
        rec["nblobs"] = len(blobs)
        rec["stored_bytes"] = sum(len(b) for b in blobs)
        self.steps[int(i)] = rec
        return rec

    def add_missing(self, i, label):
        self.steps[int(i)] = {"i": int(i), "valid_utc": label, "mode": "missing"}

    def header(self, meta):
        h = dict(meta)
        h.update({"schema": SCHEMA, "shape": list(self.shape), "tile_size": self.T, "predictor": self.predictor,
                  "prediction": PREDICTION, "residual": RESIDUAL, "palette": self.palette,
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
        """``read_range(offset, length) -> bytes``. Reads format 1 and format 2 objects."""
        self._read = read_range
        magic, hlen, ilen = HEAD.unpack(self._read(0, HEAD.size))
        if magic not in MAGICS:
            raise ExactError("not an exact-grid object")
        self.header = json.loads(self._read(HEAD.size, hlen).decode("utf-8"))
        if self.header.get("schema") != SCHEMA_OF[magic]:
            raise ExactError("unknown exact-grid schema")
        self.version = 1 if magic == MAGIC_V1 else 2
        raw = self._read(HEAD.size + hlen, ilen)
        self.index = np.frombuffer(raw, dtype="<u8").reshape(-1, 2) if ilen else np.zeros((0, 2), "<u8")
        self.data_off = HEAD.size + hlen + ilen
        self.shape = tuple(self.header["shape"])
        self.slices = list(tile_slices(self.shape, int(self.header["tile_size"])))
        self._tables = {}

    @classmethod
    def from_bytes(cls, data):
        mv = memoryview(data)
        return cls(lambda off, ln: bytes(mv[off:off + ln]))

    def blob(self, k):
        off, ln = (int(x) for x in self.index[k])
        return self._read(self.data_off + off, ln)

    def step(self, i):
        return self.header["steps"][int(i)]

    def _table(self, i, s):
        if "palette" not in s:
            return None
        if i not in self._tables:
            self._tables[i] = palette_table(s["palette"])
        return self._tables[i]

    @staticmethod
    def _shape(sl):
        return sl[0].stop - sl[0].start, sl[1].stop - sl[1].start

    def _codes(self, s, k):
        return unshuffle(_decompressor().decompress(self.blob(s["blob"] + k)), DTYPES[s["dtype"]],
                         self._shape(self.slices[k]))

    def tile(self, i, k):
        """Packed integers (uint32) of block ``k`` of step ``i`` alone, or None if the step is not ``ints``."""
        s = self.step(i)
        if s["mode"] != "ints":
            return None
        if self.version == 1:
            return unpredict_v1(self._codes(s, k), self.header["predictor"]).astype(np.uint32)
        x = unpredict(self._codes(s, k), self.header["predictor"])
        table = self._table(i, s)
        return (table[x] if table is not None else x).astype(np.uint32)

    def ints(self, i):
        """Packed integers of step i (uint32, grid shape), or None if the step is stored as the source message."""
        s = self.step(i)
        if s["mode"] != "ints":
            return None
        out = np.empty(self.shape, dtype=np.uint32)
        if self.version == 1:
            for k, sl in enumerate(self.slices):
                out[sl] = unpredict_v1(self._codes(s, k), self.header["predictor"])
            return out
        table = self._table(i, s)
        groups = {}
        for k, sl in enumerate(self.slices):
            groups.setdefault(self._shape(sl), []).append(k)
        for shp, ks in groups.items():  # blocks of one shape are inverted together (same result as one by one)
            x = unpredict(np.stack([self._codes(s, k) for k in ks], axis=-1), self.header["predictor"],
                          DTYPES[s["dtype"]], blocks_last=True)
            if table is not None:
                x = table[x]
            for n, k in enumerate(ks):
                out[self.slices[k]] = x[..., n]
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


def _verify_steps(r, ids, decode, field):
    """(checked, [failed step]) for the steps ``ids`` of reader ``r``, in that order."""
    failed, checked = [], 0
    for i in ids:
        s = r.step(i)
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
    return checked, failed


def _pread_all(fd, off, ln):
    out = bytearray()
    while len(out) < ln:
        b = os.pread(fd, ln - len(out), off + len(out))
        if not b:
            break
        out += b
    return bytes(out)


def verify_file_steps(path, ids, decode=None, field=None):
    """``_verify_steps`` on an object stored in a file (run in a worker process)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        return _verify_steps(ExactReader(lambda off, ln: _pread_all(fd, off, ln)), ids, decode, field)
    finally:
        os.close(fd)


def verify(data, decode=None, field=None, submit=None, parts=1, staging_dir=None):
    """Decode-back of a whole object: every step rebuilt from the bytes and compared bit for bit (sha256 of the
    integers and of the float64 values; ``grib`` steps: sha256 of the stored bytes, and of the values the decoder
    gives for them when ``decode`` is given). Returns (ok, {checked, failed: [step, ...]}).

    With ``submit`` (par.submit) and ``parts`` > 1 the steps are split into ``parts`` runs checked in worker
    processes, each reading the same bytes back from a file in ``staging_dir``; the result is the same."""
    r = ExactReader.from_bytes(data)
    ids = [s["i"] for s in r.header["steps"]]
    live = [i for i in ids if r.step(i)["mode"] != "missing"]
    if submit is None or parts <= 1 or len(live) <= 1:
        checked, failed = _verify_steps(r, ids, decode, field)
        return not failed, {"checked": checked, "failed": failed}
    fd, path = tempfile.mkstemp(suffix=".xgv", dir=staging_dir)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        n = min(int(parts), len(live))
        runs = [live[k * len(live) // n:(k + 1) * len(live) // n] for k in range(n)]
        futures = [submit(verify_file_steps, path, run, decode, field) for run in runs]
        checked, failed = 0, []
        for f in futures:
            c, fl = f.result()
            checked += c
            failed.extend(fl)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
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
    return {"format": header["schema"], "formula": FORMULA, "shape": header["shape"],
            "tile_size": header["tile_size"], "predictor": header["predictor"], "steps": steps}
