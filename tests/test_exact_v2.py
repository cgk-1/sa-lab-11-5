"""Exact-grid format 2: zigzag residuals under dxy / med / raw prediction, the reflectivity palette, golden bytes,
one-block reads, and format 1 objects (written by the format 1 code) still read back bit for bit."""

import hashlib
import os

import numpy as np
import pytest

from feeder import exact, grib

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
U2 = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 16, "bitmap": False}
U4 = {"template": 0, "R": 0.0, "E": 0, "D": 0, "nbits": 32, "bitmap": False}
REFL = {"template": 41, "R": -9990.0, "E": 0, "D": 1, "nbits": 16, "bitmap": False}  # X 0 = -999, 9000 = -99


def _cube(tmp_path, shape, n, predictor, palette="off", T=32, store="ints"):
    return exact.ExactCube(shape, n, str(tmp_path), T, predictor, 19, store, 2, palette)


def _round_trip(tmp_path, fields, pack, predictor, palette="off", T=32):
    shape = fields[0].shape
    cube = _cube(tmp_path, shape, len(fields), predictor, palette, T)
    recs = []
    for i, x in enumerate(fields):
        recs.append(cube.add_step(i, f"t{i}", grib.ints_to_values(x, pack), pack, b"\x00" * 64))
    data = cube.to_bytes({"layer": "t"})
    cube.close()
    ok, rep = exact.verify(data)
    assert ok and rep["checked"] == len(fields)
    r = exact.ExactReader.from_bytes(data)
    assert r.version == 2 and data[:8] == b"XGRID002" and r.header["schema"] == "exact-grid/2"
    for i, x in enumerate(fields):
        assert np.array_equal(r.ints(i), x), i
        assert grib.same_bits(r.values(i), grib.ints_to_values(x, pack))
    return data, recs, r


# ---------------------------------------------------------------- residual mapping

def test_zigzag_maps_small_negatives_to_small_codes_and_wraps():
    r = np.array([0, -1, 1, -2, 2, -32768, 32767, 65535, -65535])
    z = exact.zigzag(r, np.uint16)
    assert z.tolist() == [0, 1, 2, 3, 4, 65535, 65534, 1, 2]  # 65535 == -1 and -65535 == 1 mod 2**16
    assert exact.unzigzag(z[:7]).tolist() == r[:7].tolist()
    assert exact.zigzag(np.array([-1, 127, -128]), np.uint8).tolist() == [1, 254, 255]


def test_descending_ramp_has_small_codes_under_v2():
    t = np.tile(np.arange(4000, 3872, -1, dtype=np.uint16), (8, 1))  # every step left is -1
    assert int(exact.predict(t, "dxy")[:, 1:].max()) <= 1
    assert int(exact.predict(t, "med")[1:, 1:].max()) <= 1
    assert int(exact.predict_v1(t, "dxy")[0, 1:].min()) == 0xFFFF  # format 1 wrapped -1 to 0xFFFF


def _extreme_fields(dtype_max, shape=(70, 90)):
    i, j = np.mgrid[0:shape[0], 0:shape[1]]
    hi = dtype_max
    return [
        np.where((i + j) % 2 == 0, 0, hi).astype(np.uint32),                   # checkerboard: +-max residuals
        ((hi - (i * 97 + j * 31)) % (hi + 1)).astype(np.uint32),             # descending: negative residuals
        np.where(j < 45, hi, 0).astype(np.uint32),                           # one cliff per row
        np.full(shape, hi, np.uint32),                                       # constant at the top of the range
        ((i * 7919 + j * 104729 + i * j) % (hi + 1)).astype(np.uint32),      # scrambled
    ]


@pytest.mark.parametrize("predictor", ["dxy", "med", "raw"])
@pytest.mark.parametrize("pack,hi,dtype", [(U2, 0xFF, "u1"), (U2, 0xFFFF, "u2"), (U4, 0xFFFFFFFF, "u4")])
def test_round_trip_negative_and_extreme_residuals(tmp_path, predictor, pack, hi, dtype):
    fields = _extreme_fields(hi)
    _, recs, r = _round_trip(tmp_path, fields, pack, predictor)
    assert all(rc["mode"] == "ints" and rc["dtype"] == dtype for rc in recs[:3])
    # each block alone gives the same integers as the whole step
    for k in (0, 2, len(r.slices) - 1):
        assert np.array_equal(r.tile(0, k), fields[0][r.slices[k]])


@pytest.mark.parametrize("shape,T", [((1, 1), 128), ((1, 300), 128), ((300, 1), 128), ((129, 257), 128),
                                     ((5, 5), 1)])
@pytest.mark.parametrize("predictor", ["dxy", "med"])
def test_round_trip_odd_shapes(tmp_path, shape, T, predictor):
    rng = np.random.default_rng(3)
    fields = [rng.integers(0, 4096, shape).astype(np.uint32) for _ in range(2)]
    _round_trip(tmp_path, fields, U2, predictor, T=T)


def test_med_matches_loco_i_reference():
    rng = np.random.default_rng(9)
    x = rng.integers(0, 65536, (40, 33)).astype(np.int64)
    ref = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            a = x[i, j - 1] if j else 0
            b = x[i - 1, j] if i else 0
            c = x[i - 1, j - 1] if i and j else 0
            ref[i, j] = min(a, b) if c >= max(a, b) else (max(a, b) if c <= min(a, b) else a + b - c)
    assert np.array_equal(exact.prediction(x, "med"), ref)
    z = exact.predict(x.astype(np.uint16), "med")
    assert np.array_equal(exact.unpredict(z, "med"), x)


# ---------------------------------------------------------------- palette (reflectivity)

def _refl(shape=(96, 130), k=0):
    i, j = np.mgrid[0:shape[0], 0:shape[1]]
    x = np.full(shape, 9000, np.uint32)                      # -99: no echo
    x[:, : shape[1] // 4] = 0                                # -999: outside coverage
    echo = (i - 50) ** 2 + (j - 80) ** 2 < 900
    x[echo] = (9990 + 5 * ((i + 2 * j + k) % 150))[echo]     # 0 .. 74.5 dBZ in 0.5 dB steps
    return x


def test_palette_round_trip_with_special_codes(tmp_path):
    fields = [_refl(k=k) for k in range(3)]
    data, recs, r = _round_trip(tmp_path, fields, REFL, "med", palette="auto")
    for i, rc in enumerate(recs):
        assert rc["mode"] == "ints" and rc["dtype"] == "u1" and rc["palette"]["count"] == len(np.unique(fields[i]))
        table = exact.palette_table(r.step(i)["palette"])
        assert table[0] == 0 and 9000 in table  # -999 and -99 are table entries like any value
    v = r.values(0)
    assert (v == -999.0).sum() == (fields[0] == 0).sum() and (v == -99.0).sum() == (fields[0] == 9000).sum()
    # the palette lives in the object: a reader needs nothing else
    assert '"palette":{"b64":' in data[24:24 + 400000].decode("ascii", "replace")


def test_palette_all_missing_tile_and_all_missing_step(tmp_path):
    a = _refl()
    a[:64, :64] = 0                               # one block entirely outside coverage
    b = np.zeros((96, 130), np.uint32)            # a whole step of one code: a one-entry table
    _, recs, r = _round_trip(tmp_path, [a, b], REFL, "med", palette="auto", T=64)
    assert recs[1]["palette"]["count"] == 1
    assert np.array_equal(r.tile(0, 0), np.zeros((64, 64), np.uint32))
    assert all(np.array_equal(r.tile(1, k), np.zeros_like(b[sl])) for k, sl in enumerate(r.slices))


def test_palette_full_0_to_255_and_257_values(tmp_path):
    i, j = np.mgrid[0:64, 0:64]
    full = ((i * 64 + j) % 256).astype(np.uint32) * 5 + 9990    # exactly 256 distinct integers: u1 indices 0..255
    over = full.copy()
    over[0, 0] = 0                                               # 257 distinct: no palette, the integers as they are
    low = ((i * 64 + j) % 256).astype(np.uint32)                 # the integers 0..255 themselves
    _, recs, r = _round_trip(tmp_path, [full, over, low], REFL, "med", palette="auto")
    assert recs[0]["palette"]["count"] == 256 and recs[0]["dtype"] == "u1"
    assert "palette" not in recs[1] and recs[1]["dtype"] == "u2"
    assert recs[2]["palette"]["count"] == 256 and recs[2]["palette"]["dtype"] == "u1"
    assert np.array_equal(exact.palette_table(r.step(0)["palette"]), np.unique(full))


def test_palette_off_keeps_the_integers(tmp_path):
    _, recs, _ = _round_trip(tmp_path, [_refl()], REFL, "med", palette="off")
    assert "palette" not in recs[0] and recs[0]["dtype"] == "u2"


# ---------------------------------------------------------------- store: auto, corruption, one-block reads

def test_auto_still_keeps_the_source_when_smaller(tmp_path):
    rng = np.random.default_rng(2)
    x = rng.integers(0, 4096, (64, 64)).astype(np.uint32)
    cube = _cube(tmp_path, (64, 64), 2, "med", store="auto")
    v = grib.ints_to_values(x, U2)
    assert cube.add_step(0, "a", v, U2, b"\x00" * 100000)["mode"] == "ints"
    assert cube.add_step(1, "b", v, U2, b"tiny")["mode"] == "grib"
    data = cube.to_bytes({})
    cube.close()
    assert exact.verify(data)[0] and exact.ExactReader.from_bytes(data).source(1) == b"tiny"


@pytest.mark.parametrize("predictor,palette", [("dxy", "off"), ("med", "off"), ("med", "auto")])
def test_corruption_is_caught(tmp_path, predictor, palette):
    data, _, _ = _round_trip(tmp_path, [_refl(), _refl(k=1)], REFL, predictor, palette)
    bad = bytearray(data)
    bad[-5] ^= 0x10
    ok, rep = exact.verify(bytes(bad))
    assert not ok and rep["failed"] == [1]


def test_one_block_needs_only_its_own_range_reads(tmp_path):
    data, _, _ = _round_trip(tmp_path, [_refl(), _refl(k=1)], REFL, "med", "auto")
    reads = []

    def rr(off, ln):
        reads.append((off, ln))
        return data[off:off + ln]

    r = exact.ExactReader(rr)
    head = sum(ln for _, ln in reads)
    k = 5
    t = r.tile(1, k)
    assert np.array_equal(t, _refl(k=1)[r.slices[k]])
    assert len(reads) == 4 and reads[-1][1] == int(r.index[r.step(1)["blob"] + k][1])  # head, header, index, blob
    assert head < len(data)


# ---------------------------------------------------------------- golden bytes of the new encoding

def _golden_object(tmp_path, predictor, palette, kind):
    shape = (70, 90)
    i, j = np.mgrid[0:shape[0], 0:shape[1]]
    if kind == "smooth":  # integer arithmetic only, so the field is the same on every machine
        fields = [(1500 + 11 * i + 7 * j - (i * j) // 20 + ((i * 7 + j * 3 + k) % 5)).astype(np.uint32)
                  for k in range(2)]
        pack = U2
    else:
        fields = [_refl(shape, k) for k in range(2)]
        pack = REFL
    cube = exact.ExactCube(shape, 3, str(tmp_path), 32, predictor, 19, "ints", 1, palette)
    for k, x in enumerate(fields):
        cube.add_step(k, f"2024-05-07T0{k}:00:00Z", grib.ints_to_values(x, pack), pack, b"source-%d" % k)
    data = cube.to_bytes({"dataset": "golden"})
    cube.close()
    return data


# sha256 of whole format 2 objects (zstandard 0.25.0 as pinned in requirements.lock; another zstd version may give
# other, equally lossless, bytes: then these must be re-derived on purpose, never silently)
GOLDEN_V2 = {
    ("dxy", "off", "smooth"): "0ec44da3e526e0ef76e1f83f97305cdd0adaa4841423d94d93b7aa965ee5e34b",
    ("med", "off", "smooth"): "23b65b1d8ea3fb99120928a1691b10ad29390b3133ffbdea8b145e1e2988be7d",
    ("med", "auto", "refl"): "0dbb8a92524f65303eee9bb34b601624bb5c0b0656ff846e5c8167a90a4edcb9",
    ("raw", "off", "refl"): "631eba4f5346d1fa20adb2b7beb32d4d24721dd3c737a77aa4af454bddd5de97",
}


@pytest.mark.parametrize("key", sorted(GOLDEN_V2))
def test_golden_sha256_of_the_new_encoding(tmp_path, key):
    data = _golden_object(tmp_path, *key)
    assert exact.verify(data)[0]
    assert hashlib.sha256(data).hexdigest() == GOLDEN_V2[key]


# ---------------------------------------------------------------- format 1 objects stay readable

def _v1_field(kind, shape, k):
    """The fields the format 1 fixtures were built from (with the format 1 code, git ca5e226). The "smooth" one came
    from float sin / cos (not rebuilt here: libm may differ by machine); it is checked by its integer sha256."""
    i, j = np.mgrid[0:shape[0], 0:shape[1]]
    if kind == "smooth":
        return None
    if kind == "wide":
        return ((i * 104729 + j * 7919 + k * 31337) % 90001).astype(np.uint32)
    x = np.full(shape, 9000, np.uint32)
    x[:, : shape[1] // 3] = 0
    m = (i - 30) ** 2 + (j - 50) ** 2 < 200
    x[m] = (9990 + 5 * ((i + j + k) % 120))[m]
    return x


# file -> (field kind, predictor, sha256 of the object, sha256 of the integers of steps 0 and 1)
V1_FIXTURES = {
    "xgrid-v1-dxy.bin": ("smooth", "dxy", "ab645643c65c03be85a8806ba9fc0a63999c08a6e3e416dd5d64624cbdfd4e1a",
                         ("34a6ad2dfcb00f069a917a1547d4272d350dc3559454f26a4a769bf6d70b30d5",
                          "1beb7737e062ae4a24274e27f1ec3bc67f9192bcb889672fac50485cc4a67480")),
    "xgrid-v1-raw.bin": ("refl", "raw", "e8f2cab53965c06ea48b5e28519957cc818ddfeedea286b576d9f119ecaecc9f",
                         ("7c8af2f3bf7e14b6c54399a9697098eb9ef3f0adbafa9bffa358b096e1bcaefe",
                          "7f2421d75daafdd0058ccb21c7d13f45614085f1e83cb447d84ed09cf29c8e6a")),
    "xgrid-v1-wide.bin": ("wide", "dxy", "a8392031d5373354f971d669a2e6ca2184e5cd28c67534bd7c772eadafb49a33",
                          ("10ced8c6d85770bae8ea3b351d21f8aa68c363ab39cf1ddc174d2a2261466eca",
                           "dc7e9c9c28e0a1e22aa888d09063bfb4970950e5aeeb282460578ac70882804d")),
}


@pytest.mark.parametrize("name", sorted(V1_FIXTURES))
def test_format_1_objects_still_read_bit_for_bit(name):
    kind, predictor, sha, ints_sha = V1_FIXTURES[name]
    with open(os.path.join(FIX, name), "rb") as fh:
        data = fh.read()
    assert hashlib.sha256(data).hexdigest() == sha and data[:8] == b"XGRID001"
    ok, rep = exact.verify(data)
    assert ok and rep["checked"] == 3
    r = exact.ExactReader.from_bytes(data)
    assert r.version == 1 and r.header["schema"] == "exact-grid/1" and r.header["predictor"] == predictor
    shape = tuple(r.header["shape"])
    for k in (0, 1):
        got = r.ints(k)
        assert exact.ints_sha256(got) == ints_sha[k] == r.step(k)["ints_sha256"]
        assert np.array_equal(r.tile(k, 4), got[r.slices[4]])
        x = _v1_field(kind, shape, k)
        if x is not None:
            assert np.array_equal(got, x)
    assert r.step(2)["mode"] == "missing" and r.values(2) is None
    assert r.source(3) == b"source-message-3"
    assert exact.manifest_record(r.header)["format"] == "exact-grid/1"
