"""Display tiles: u8-coded native-grid rasters in PMTiles v3 archives, with an exact overflow sidecar.

Encoding (per layer, from config ``encodings``):
- x = value * factor; code = rint((x - offset) / step) for 0 <= code < first reserved code;
- ``specials`` map exact source values (e.g. a missing-data flag) to reserved codes; NaN -> ``nan_code``;
- code 255 = overflow: the exact source value is in the sidecar (step, cell, float64 value).

Layout: the grid is cut into ``tile_size`` x ``tile_size`` cell blocks on its native grid (never reprojected). Each
block is one 8-bit greyscale PNG "time atlas": all steps stacked vertically (step s = rows s*T .. s*T+T-1). Block
(bx, by) is stored at PMTiles (z, x=bx, y=by) with z = ceil(log2(max(blocks across, blocks down))). Rows follow the
source's scan order. Grid geometry, steps and the encoding go in the PMTiles JSON metadata.

Written from the public PNG and PMTiles v3 specifications.
"""

import gzip
import io
import json
import math
import os
import struct
import tempfile
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

OVERFLOW = 255
TILE_TYPE_PNG = 2
COMPRESSION_NONE = 1
COMPRESSION_GZIP = 2


# ---------------------------------------------------------------- encoding

class Encoding:
    def __init__(self, spec):
        self.factor = float(spec["factor"])
        self.offset = float(spec["offset"])
        self.step = float(spec["step"])
        self.specials = {float(k): int(v) for k, v in spec.get("specials", {}).items()}
        self.nan_code = int(spec.get("nan_code", 254))
        reserved = [OVERFLOW, self.nan_code, *self.specials.values()]
        self.n_numeric = min(reserved)
        self.spec = dict(spec)

    def encode(self, values):
        """values (float array, source units) -> (codes u8, overflow mask)."""
        v = np.asarray(values, dtype=np.float64)
        x = v * self.factor
        with np.errstate(invalid="ignore"):
            c = np.rint((x - self.offset) / self.step)
        nan = np.isnan(v)
        numeric = ~nan & (c >= 0) & (c < self.n_numeric)
        codes = np.full(v.shape, OVERFLOW, dtype=np.uint8)
        codes[numeric] = c[numeric].astype(np.uint8)
        codes[nan] = self.nan_code
        for sval, scode in self.specials.items():
            m = v == sval
            codes[m] = scode
        overflow = codes == OVERFLOW
        return codes, overflow

    def decode(self, codes):
        c = np.asarray(codes)
        out = np.full(c.shape, np.nan)
        num = c < self.n_numeric
        out[num] = (c[num].astype(np.float64) * self.step + self.offset) / self.factor
        for sval, scode in self.specials.items():
            out[c == scode] = sval
        return out

    def max_error(self):
        """Largest allowed |decoded - source| for numeric codes, in source units."""
        return self.step / 2.0 / self.factor + 1e-9


# ---------------------------------------------------------------- PNG (8-bit greyscale)

def _chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def png_encode(img, level=6):
    img = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = img.shape
    prev = np.vstack([np.zeros((1, w), np.uint8), img[:-1]])
    filt = (img.astype(np.int16) - prev.astype(np.int16)).astype(np.uint8)  # "Up" filter, mod 256
    raw = np.hstack([np.full((h, 1), 2, np.uint8), filt]).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, level)) + \
        _chunk(b"IEND", b"")


def png_decode(data):
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, w = 8, [], None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + n]
        if zlib.crc32(tag + body) & 0xFFFFFFFF != struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0]:
            raise ValueError("PNG CRC mismatch")
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if (depth, ctype, interlace) != (8, 0, 0):
                raise ValueError("unsupported PNG format")
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
        pos += 12 + n
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8).reshape(h, w + 1)
    if np.all(raw[:, 0] == 2):  # every row "Up": a running sum down the columns, mod 256
        return np.cumsum(raw[:, 1:], axis=0, dtype=np.uint8)
    out = np.empty((h, w), np.uint8)
    prev = np.zeros(w, np.uint8)
    for r in range(h):
        f, row = raw[r, 0], raw[r, 1:]
        if f == 0:
            cur = row
        elif f == 2:
            cur = (row.astype(np.uint16) + prev).astype(np.uint8)
        elif f == 1:
            cur = np.frombuffer(np.cumsum(row, dtype=np.uint64).astype(np.uint8).tobytes(), np.uint8)
        else:
            raise ValueError(f"PNG filter {f} not supported")
        out[r] = cur
        prev = cur
    return out


# ---------------------------------------------------------------- PMTiles v3

def zxy_to_tileid(z, x, y):
    if z > 26:
        raise ValueError("zoom too large")
    if x >= (1 << z) or y >= (1 << z):
        raise ValueError("tile outside zoom level")
    acc = ((1 << (2 * z)) - 1) // 3
    d = 0
    s = 1 << (z - 1) if z > 0 else 0
    tx, ty = x, y
    while s > 0:
        rx = 1 if tx & s else 0
        ry = 1 if ty & s else 0
        d += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                tx = s - 1 - tx
                ty = s - 1 - ty
            tx, ty = ty, tx
        s >>= 1
    return acc + d


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf, pos):
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _serialize_dir(entries):
    out = bytearray(_varint(len(entries)))
    last = 0
    for e in entries:
        out += _varint(e[0] - last)
        last = e[0]
    for e in entries:
        out += _varint(e[3])
    for e in entries:
        out += _varint(e[2])
    for i, e in enumerate(entries):
        if i > 0 and e[1] == entries[i - 1][1] + entries[i - 1][2]:
            out += _varint(0)
        else:
            out += _varint(e[1] + 1)
    return gzip.compress(bytes(out), mtime=0)


def _deserialize_dir(data):
    buf = gzip.decompress(data)
    n, pos = _read_varint(buf, 0)
    ids, runs, lens, offs = [], [], [], []
    last = 0
    for _ in range(n):
        v, pos = _read_varint(buf, pos)
        last += v
        ids.append(last)
    for _ in range(n):
        v, pos = _read_varint(buf, pos)
        runs.append(v)
    for _ in range(n):
        v, pos = _read_varint(buf, pos)
        lens.append(v)
    for i in range(n):
        v, pos = _read_varint(buf, pos)
        offs.append(offs[i - 1] + lens[i - 1] if (v == 0 and i > 0) else v - 1)
    return [(ids[i], offs[i], lens[i], runs[i]) for i in range(n)]


def write_pmtiles(out_path, tiles, metadata, min_zoom, max_zoom, bounds=None, tile_type=TILE_TYPE_PNG):
    """``tiles``: iterable of (tile_id, bytes) in increasing tile_id order. Identical tile contents are stored once."""
    data_fh = tempfile.TemporaryFile()
    entries, seen, offset, addressed = [], {}, 0, 0
    last_id = -1
    for tid, content in tiles:
        if tid <= last_id:
            raise ValueError("tiles must be in increasing tile id order")
        last_id = tid
        addressed += 1
        key = (len(content), zlib.crc32(content), hash(content))
        if key in seen:
            off, ln = seen[key]
        else:
            off, ln = offset, len(content)
            data_fh.write(content)
            offset += ln
            seen[key] = (off, ln)
        if entries and entries[-1][1] == off and entries[-1][0] + entries[-1][3] == tid:
            e = entries[-1]
            entries[-1] = (e[0], e[1], e[2], e[3] + 1)
        else:
            entries.append((tid, off, ln, 1))
    root = _serialize_dir(entries)
    leaves = b""
    if len(root) > 16384 - 127:
        root_entries, leaf_parts, leaf_off = [], [], 0
        size = 4096
        for i in range(0, len(entries), size):
            chunk = _serialize_dir(entries[i:i + size])
            root_entries.append((entries[i][0], leaf_off, len(chunk), 0))
            leaf_parts.append(chunk)
            leaf_off += len(chunk)
        root = _serialize_dir(root_entries)
        leaves = b"".join(leaf_parts)
    meta = gzip.compress(json.dumps(metadata, sort_keys=True).encode("utf-8"), mtime=0)
    root_off = 127
    meta_off = root_off + len(root)
    leaf_off = meta_off + len(meta)
    data_off = leaf_off + len(leaves)
    b = bounds or (-180.0, -85.0, 180.0, 85.0)
    e7 = lambda v: int(round(v * 1e7))  # noqa: E731
    header = b"PMTiles" + bytes([3]) + struct.pack(
        "<QQQQQQQQQQQ", root_off, len(root), meta_off, len(meta), leaf_off, len(leaves), data_off, offset,
        addressed, len(entries), len(seen)) + struct.pack(
        "<BBBBBB", 1, COMPRESSION_GZIP, COMPRESSION_NONE, tile_type, min_zoom, max_zoom) + struct.pack(
        "<iiii", e7(b[0]), e7(b[1]), e7(b[2]), e7(b[3])) + struct.pack(
        "<Bii", min_zoom, e7((b[0] + b[2]) / 2), e7((b[1] + b[3]) / 2))
    assert len(header) == 127
    with open(out_path, "wb") as out:
        out.write(header + root + meta + leaves)
        data_fh.seek(0)
        while True:
            chunk = data_fh.read(1 << 22)
            if not chunk:
                break
            out.write(chunk)
    data_fh.close()
    return {"addressed": addressed, "entries": len(entries), "contents": len(seen)}


class PMTilesReader:
    def __init__(self, read_range):
        """``read_range(offset, length) -> bytes``."""
        self._read = read_range
        h = self._read(0, 127)
        if h[:7] != b"PMTiles" or h[7] != 3:
            raise ValueError("not a PMTiles v3 archive")
        f = struct.unpack("<QQQQQQQQQQQ", h[8:96])
        (self.root_off, self.root_len, self.meta_off, self.meta_len, self.leaf_off, self.leaf_len,
         self.data_off, self.data_len, self.addressed, self.n_entries, self.n_contents) = f
        self.clustered, self.internal_comp, self.tile_comp, self.tile_type, self.min_zoom, self.max_zoom = \
            struct.unpack("<BBBBBB", h[96:102])
        self._root = _deserialize_dir(self._read(self.root_off, self.root_len))

    @classmethod
    def from_bytes(cls, data):
        return cls(lambda off, ln: data[off:off + ln])

    def metadata(self):
        return json.loads(gzip.decompress(self._read(self.meta_off, self.meta_len)).decode("utf-8"))

    def _find(self, entries, tid):
        lo, hi = 0, len(entries) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if entries[mid][0] <= tid:
                best = entries[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def get(self, z, x, y):
        tid = zxy_to_tileid(z, x, y)
        entries = self._root
        for _ in range(4):
            e = self._find(entries, tid)
            if e is None:
                return None
            if e[3] == 0:
                entries = _deserialize_dir(self._read(self.leaf_off + e[1], e[2]))
                continue
            if tid < e[0] + e[3]:
                return self._read(self.data_off + e[1], e[2])
            return None
        return None


# ---------------------------------------------------------------- tile cube

class TileCube:
    def __init__(self, shape, n_steps, encoding, staging_dir, tile_size=128, step_labels=None):
        self.shape = tuple(shape)
        self.n_steps = int(n_steps)
        self.enc = encoding
        self.T = int(tile_size)
        self.step_labels = list(step_labels or range(self.n_steps))
        fd, self._path = tempfile.mkstemp(suffix=".u8", dir=staging_dir)
        os.close(fd)
        self.codes = np.lib.format.open_memmap(self._path, mode="w+", dtype=np.uint8,
                                               shape=(self.n_steps, *self.shape))
        self.codes[:] = encoding.nan_code
        self.sidecar = {}
        self.present = []
        self.decode_back_ok = True

    def close(self):
        try:
            del self.codes
            os.unlink(self._path)
        except OSError:
            pass

    def add_step(self, i, values):
        v = np.asarray(values, dtype=np.float64)
        if v.shape != self.shape:
            raise ValueError("step shape does not match the cube")
        codes, overflow = self.enc.encode(v)
        cells = np.nonzero(overflow.ravel())[0]
        if cells.size:
            self.sidecar[int(i)] = (cells.astype(np.uint32), v.ravel()[cells].astype(np.float64))
        self.codes[i] = codes
        # decode-back on the step: numeric codes within half a step, specials / NaN / sidecar exact
        dec = self.enc.decode(codes).ravel()
        if cells.size:
            dec[cells] = self.sidecar[int(i)][1]
        src = v.ravel()
        both_nan = np.isnan(dec) & np.isnan(src)
        ok = both_nan | (np.abs(dec - src) <= self.enc.max_error())
        if not bool(np.all(ok)):
            self.decode_back_ok = False
        self.present.append(int(i))

    def decoded_step(self, i):
        out = self.enc.decode(self.codes[i]).ravel()
        if int(i) in self.sidecar:
            cells, vals = self.sidecar[int(i)]
            out[cells] = vals
        return out.reshape(self.shape)

    def overview(self, factor, staging_dir):
        ny, nx = self.shape
        oy, ox = math.ceil(ny / factor), math.ceil(nx / factor)
        ov = TileCube((oy, ox), self.n_steps, self.enc, staging_dir, self.T, self.step_labels)
        for i in self.present:
            d = self.decoded_step(i)
            pad = np.full((oy * factor, ox * factor), np.nan)
            pad[:ny, :nx] = d
            blocks = pad.reshape(oy, factor, ox, factor)
            with np.errstate(all="ignore"):
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    m = np.nanmax(blocks, axis=(1, 3))
            ov.add_step(i, m)
        return ov

    def blocks(self):
        ny, nx = self.shape
        return math.ceil(nx / self.T), math.ceil(ny / self.T)

    def zoom(self):
        bx, by = self.blocks()
        return max(0, math.ceil(math.log2(max(bx, by, 1))))

    def _tile_image(self, bx, by):
        T = self.T
        ny, nx = self.shape
        img = np.full((self.n_steps, T, T), self.enc.nan_code, np.uint8)
        ys, xs = by * T, bx * T
        part = self.codes[:, ys:min(ys + T, ny), xs:min(xs + T, nx)]
        img[:, :part.shape[1], :part.shape[2]] = part
        return img.reshape(self.n_steps * T, T)

    def sidecar_parquet(self):
        steps, cells, vals = [], [], []
        for s in sorted(self.sidecar):
            c, v = self.sidecar[s]
            steps.append(np.full(c.size, s, np.uint16))
            cells.append(c)
            vals.append(v)
        if not steps:
            return None
        t = pa.table({"step": pa.array(np.concatenate(steps), pa.uint16()),
                      "cell": pa.array(np.concatenate(cells), pa.uint32()),
                      "value": pa.array(np.concatenate(vals), pa.float64())})
        buf = io.BytesIO()
        pq.write_table(t, buf, compression="zstd")
        return buf.getvalue()

    def write(self, out_path, metadata, bounds=None):
        z = self.zoom()
        bx, by = self.blocks()
        order = sorted(((zxy_to_tileid(z, x, y), x, y) for y in range(by) for x in range(bx)))
        meta = dict(metadata)
        meta.update({"grid_shape": list(self.shape), "tile_size": self.T, "steps": self.n_steps,
                     "step_labels": self.step_labels, "steps_present": self.present, "zoom": z,
                     "blocks": [bx, by], "encoding": self.enc.spec, "overflow_code": OVERFLOW,
                     "layout": "time atlas: step s = PNG rows s*T..s*T+T-1; source scan order"})
        stats = write_pmtiles(out_path, ((tid, png_encode(self._tile_image(x, y))) for tid, x, y in order),
                              meta, z, z, bounds)
        return stats

    def verify_file(self, path):
        """Reads every tile back from the written archive and compares the codes exactly."""
        with open(path, "rb") as fh:
            data = fh.read()
        r = PMTilesReader.from_bytes(data)
        z = r.min_zoom
        bx, by = self.blocks()
        for y in range(by):
            for x in range(bx):
                t = r.get(z, x, y)
                if t is None or not np.array_equal(png_decode(t), self._tile_image(x, y)):
                    return False
        return True
