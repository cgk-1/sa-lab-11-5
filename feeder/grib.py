"""GRIB2: a pure-Python section walker (for files without an .idx), pygrib decoding with parameter-id and grid-shape
checks, and the native-integer view of a decoded field (the packed integers and the R / E / D factors that rebuild the
decoded values bit for bit)."""

import struct

import numpy as np


class GribError(Exception):
    pass


# ---------------------------------------------------------------- header walk (no decoding library needed)

def parse_section0(buf):
    """(discipline, edition, total_length) from the first 16 bytes of a message."""
    if len(buf) < 16 or buf[:4] != b"GRIB":
        raise GribError("not a GRIB message start")
    discipline, edition = buf[6], buf[7]
    if edition != 2:
        raise GribError(f"GRIB edition {edition} not supported")
    total = struct.unpack(">Q", buf[8:16])[0]
    return discipline, edition, total


def _scaled(scale_byte, value):
    if value == 0xFFFFFFFF or scale_byte == 0xFF:
        return None
    scale = scale_byte - 256 if scale_byte > 127 else scale_byte
    return value / (10 ** scale)


def parse_header(buf):
    """Parses sections 0-4 of one GRIB2 message from ``buf`` (which starts at the message).

    Returns a dict, or raises NeedMore(n) when ``buf`` is shorter than the headers."""
    discipline, _, total = parse_section0(buf)
    pos = 16
    out = {"discipline": discipline, "length": total}
    while True:
        if pos + 5 > len(buf):
            raise NeedMore(pos + 5)
        if buf[pos:pos + 4] == b"7777":
            raise GribError("end of message before section 4")
        slen, snum = struct.unpack(">IB", buf[pos:pos + 5])
        if slen < 5:
            raise GribError("bad section length")
        if snum == 3:
            if pos + 14 > len(buf):
                raise NeedMore(pos + 14)
            out["grid_template"] = struct.unpack(">H", buf[pos + 12:pos + 14])[0]
            out["n_points"] = struct.unpack(">I", buf[pos + 6:pos + 10])[0]
        if snum == 4:
            if pos + 28 > len(buf):
                raise NeedMore(pos + 28)
            s = buf[pos:pos + 28]
            out["template"] = struct.unpack(">H", s[7:9])[0]
            out["category"] = s[9]
            out["number"] = s[10]
            out["surface_type"] = s[22]
            out["surface_value"] = _scaled(s[23], struct.unpack(">I", s[24:28])[0])
            return out
        pos += slen


class NeedMore(GribError):
    def __init__(self, n):
        super().__init__(f"need {n} bytes")
        self.n = n


def matches(header, field):
    """True if a parsed header is the field described by config (``grib`` = [discipline, category, number],
    optional ``walk`` = {"surface": [type, value], "template": n})."""
    d, c, n = field["grib"]
    if (header["discipline"], header["category"], header["number"]) != (d, c, n):
        return False
    walk = field.get("walk") or {}
    if "surface" in walk:
        st, sv = walk["surface"]
        if header["surface_type"] != st or header["surface_value"] is None or abs(header["surface_value"] - sv) > 1e-6:
            return False
    if "template" in walk and header.get("template") != walk["template"]:
        return False
    return True


# ---------------------------------------------------------------- decoding

class Grid:
    """Grid geometry of a decoded message: cell lookup for points and cell lat/lon arrays."""

    def __init__(self, shape, latlons_fn=None, locate_fn=None, signature=None):
        self.shape = tuple(shape)
        self._latlons_fn = latlons_fn
        self._locate_fn = locate_fn
        self._latlons = None
        self.signature = signature

    def latlons(self):
        if self._latlons is None:
            if self._latlons_fn is None:
                raise GribError("grid has no coordinates")
            lats, lons = self._latlons_fn()
            lons = np.where(lons > 180.0, lons - 360.0, lons)
            self._latlons = (np.asarray(lats, dtype=np.float64), np.asarray(lons, dtype=np.float64))
        return self._latlons

    def locate(self, lat, lon):
        """Flat cell index of the grid point nearest (lat, lon), or None outside the grid."""
        if self._locate_fn is None:
            raise GribError("grid cannot locate points")
        return self._locate_fn(lat, lon)


class Field:
    def __init__(self, values, grid, ids, packing=None):
        self.values = values
        self.grid = grid
        self.ids = ids
        self.packing = packing  # {template, R, E, D, nbits, bitmap} or None when unknown


# ---------------------------------------------------------------- native integers (exact storage)

def power(base, n):
    """base**n by repeated multiplication / division, the way the GRIB decoder builds its scale factors, so values
    rebuilt from integers match the decoder's doubles bit for bit."""
    out = 1.0
    if n < 0:
        for _ in range(-n):
            out /= base
    else:
        for _ in range(n):
            out *= base
    return out


def ints_to_values(ints, packing, out=None):
    """Decoded value of each packed integer X: ((X * 2**E) + R) * 10**-D, in float64, as the decoder computes it.

    The same three float64 operations, in the same order, done in one buffer (a fresh one, or ``out``) instead of
    one temporary array per operation: identical results, a fraction of the memory traffic on large grids."""
    s = power(2.0, int(packing["E"]))
    d = power(10.0, -int(packing["D"]))
    ints = np.asarray(ints)
    if out is None:
        out = np.empty(ints.shape, dtype=np.float64)
    np.copyto(out, ints, casting="unsafe")
    np.multiply(out, s, out=out)
    np.add(out, float(packing["R"]), out=out)
    np.multiply(out, d, out=out)
    return out


def same_bits(a, b):
    """True if two float64 arrays are identical bit for bit (NaN payloads and signed zeros included)."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.int64), b.view(np.int64))


def values_to_ints(values, packing):
    """The non-negative packed integers behind ``values`` (uint32), or None if they cannot be recovered exactly
    (no packing description, a bitmap, a negative or out-of-range integer, or a value that the formula does not
    reproduce bit for bit)."""
    if not packing or packing.get("bitmap"):
        return None
    v = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return None
    s = power(2.0, int(packing["E"]))
    d = power(10.0, -int(packing["D"]))
    # rint((v / d - R) / s), one operation at a time in one buffer (bit-identical to the expression form)
    x = np.divide(v, d)
    np.subtract(x, float(packing["R"]), out=x)
    np.divide(x, s, out=x)
    np.rint(x, out=x)
    if x.size and (x.min() < 0 or x.max() > 0xFFFFFFFF):
        return None
    ints = x.astype(np.uint32)
    if not same_bits(ints_to_values(ints, packing, out=x), v):  # x is not needed any more: reuse its buffer
        return None
    return ints


def _pygrib():
    try:
        import pygrib  # noqa: WPS433 (optional import; tests use a fake decoder)
    except ImportError as err:  # pragma: no cover
        raise GribError("pygrib is not installed") from err
    return pygrib


def _make_locator(msg, ny, nx):
    gtype = msg["gridType"]
    lat0 = float(msg["latitudeOfFirstGridPointInDegrees"])
    lon0 = float(msg["longitudeOfFirstGridPointInDegrees"])
    i_neg = bool(msg["iScansNegatively"])
    j_pos = bool(msg["jScansPositively"])
    if gtype == "lambert":
        import pyproj

        proj = pyproj.Proj(msg.projparams)
        x0, y0 = proj(lon0, lat0)
        dx, dy = float(msg["DxInMetres"]), float(msg["DyInMetres"])

        def to_xy(lat, lon):
            x, y = proj(lon, lat)
            return (x - x0) / dx, (y - y0) / dy
    elif gtype == "regular_ll":
        dx = float(msg["iDirectionIncrementInDegrees"])
        dy = float(msg["jDirectionIncrementInDegrees"])

        def to_xy(lat, lon):
            lon_ = lon % 360.0
            return ((lon_ - (lon0 % 360.0)) / dx), (lat - lat0) / dy
    else:
        return None

    def locate(lat, lon):
        fx, fy = to_xy(lat, lon)
        i = int(round(-fx if i_neg else fx))
        j = int(round(fy if j_pos else -fy))
        if 0 <= i < nx and 0 <= j < ny:
            return j * nx + i
        return None

    return locate


def decode(message, field, expect_shape=None):
    """Decodes one GRIB2 message (bytes) into a Field; checks discipline/category/number and the grid shape."""
    pygrib = _pygrib()
    try:
        msg = pygrib.fromstring(bytes(message))
    except Exception as err:
        raise GribError("message does not decode") from err
    ids = (int(msg["discipline"]), int(msg["parameterCategory"]), int(msg["parameterNumber"]))
    if field.get("grib") is not None and ids != tuple(field["grib"]):
        raise GribError(f"parameter ids {ids} != expected {tuple(field['grib'])}")
    vals = msg.values
    if np.ma.isMaskedArray(vals):
        vals = np.ma.filled(vals.astype(np.float64), np.nan)
    vals = np.asarray(vals, dtype=np.float64)
    if expect_shape is not None and tuple(vals.shape) != tuple(expect_shape):
        raise GribError(f"grid shape {vals.shape} != expected {tuple(expect_shape)}")
    ny, nx = vals.shape
    packing = _packing(msg)
    sig = (msg["gridType"], ny, nx, round(float(msg["latitudeOfFirstGridPointInDegrees"]), 5),
           round(float(msg["longitudeOfFirstGridPointInDegrees"]), 5))
    grid = Grid(vals.shape, latlons_fn=msg.latlons, locate_fn=_make_locator(msg, ny, nx), signature=sig)
    return Field(vals, grid, ids, packing)


def _packing(msg):
    try:
        return {"template": int(msg["dataRepresentationTemplateNumber"]), "R": float(msg["referenceValue"]),
                "E": int(msg["binaryScaleFactor"]), "D": int(msg["decimalScaleFactor"]),
                "nbits": int(msg["bitsPerValue"]), "bitmap": bool(int(msg["bitmapPresent"]))}
    except Exception:  # a key the decoder does not expose: exact storage falls back to the original message
        return None
