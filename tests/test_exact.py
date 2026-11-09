"""Exact grids, lossless sparse cubes, and the MESH display encoding."""

import gzip
import hashlib

import numpy as np
import pytest

from feeder import exact, grib, load_config, sparse, tile


def _field(rng, shape, pack, xmax):
    x = rng.integers(0, xmax + 1, size=shape).astype(np.uint32)
    return x, grib.ints_to_values(x, pack)


@pytest.mark.parametrize("pack,xmax,dtype", [
    ({"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12}, 4095, "u2"),             # URMA-style 0.01 m/s
    ({"template": 0, "R": 2131.0, "E": 0, "D": 1, "nbits": 12}, 3600, "u2"),          # wind direction, 0.1 deg
    ({"template": 3, "R": 0.09325958788394928, "E": -4, "D": 0, "nbits": 9}, 511, "u2"),  # HRRR 1/16 m/s
    ({"template": 41, "R": -30.0, "E": 0, "D": 1, "nbits": 16}, 200, "u1"),            # MESH, mostly -3 / -1
    ({"template": 41, "R": -9990.0, "E": 0, "D": 1, "nbits": 32}, 90000, "u4"),        # wide integers
])
@pytest.mark.parametrize("predictor", ["dxy", "raw"])
def test_ints_round_trip_bit_exact(tmp_path, pack, xmax, dtype, predictor):
    rng = np.random.default_rng(5)
    shape = (300, 270)  # edge blocks smaller than the tile
    cube = exact.ExactCube(shape, 3, str(tmp_path), 128, predictor, 3, "ints", 2)
    ref = []
    for i in range(3):
        x, v = _field(rng, shape, pack, xmax)
        rec = cube.add_step(i, f"t{i}", v, {**pack, "bitmap": False}, b"src-%d" % i)
        assert rec["mode"] == "ints" and rec["int_exact"] and rec["dtype"] == dtype
        ref.append((x, v))
    data = cube.to_bytes({"layer": "t"})
    cube.close()
    ok, rep = exact.verify(data)
    assert ok and rep["checked"] == 3
    r = exact.ExactReader.from_bytes(data)
    for i, (x, v) in enumerate(ref):
        assert np.array_equal(r.ints(i), x)
        assert grib.same_bits(r.values(i), v)


def test_values_to_ints_refuses_inexact():
    pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
    v = grib.ints_to_values(np.array([[1, 2], [3, 4]], np.uint32), pack)
    assert grib.values_to_ints(v, pack) is not None
    w = v.copy()
    w[0, 0] = np.nextafter(w[0, 0], 1.0)  # one ulp off: no integer reproduces it
    assert grib.values_to_ints(w, pack) is None
    assert grib.values_to_ints(v, {**pack, "bitmap": True}) is None
    assert grib.values_to_ints(v, None) is None


def test_auto_keeps_the_smaller_and_falls_back_to_source(tmp_path):
    pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
    flat = grib.ints_to_values(np.full((64, 64), 7, np.uint32), pack)
    cube = exact.ExactCube((64, 64), 3, str(tmp_path), 32, "dxy", 3, "auto", 1)
    big = bytes(range(256)) * 64
    assert cube.add_step(0, "a", flat, pack, big)["mode"] == "ints"            # integers far smaller
    assert cube.add_step(1, "b", flat, pack, b"tiny")["mode"] == "grib"        # source smaller: kept as is
    noisy = flat.copy()
    noisy[0, 0] += 1e-9
    rec = cube.add_step(2, "c", noisy, pack, gzip.compress(b"message", mtime=0), "gzip")
    assert rec["mode"] == "grib" and not rec["int_exact"]                       # not reproducible: source bytes
    data = cube.to_bytes({})
    cube.close()
    r = exact.ExactReader.from_bytes(data)
    assert r.source(1) == b"tiny" and gzip.decompress(r.source(2)) == b"message"
    assert exact.verify(data)[0]
    assert cube.summary() == {"steps_ints": 1, "steps_grib": 2, "steps_missing": 0}


def test_reader_needs_only_range_reads(tmp_path):
    pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
    rng = np.random.default_rng(1)
    x, v = _field(rng, (100, 100), pack, 900)
    cube = exact.ExactCube((100, 100), 2, str(tmp_path), 50, "dxy", 3, "ints", 1)
    cube.add_step(1, "b", v, pack, b"s")
    data = cube.to_bytes({"k": 1})
    cube.close()
    reads = []

    def rr(off, ln):
        reads.append((off, ln))
        return data[off:off + ln]

    r = exact.ExactReader(rr)
    assert r.step(0)["mode"] == "missing" and r.values(0) is None
    assert np.array_equal(r.ints(1), x)
    assert sum(ln for _, ln in reads) == len(data)  # header, index and each blob once; never more


def test_manifest_record_carries_factors_and_checksums(tmp_path):
    pack = {"template": 3, "R": 0.5, "E": -4, "D": 0, "nbits": 9, "bitmap": False}
    x = np.arange(64, dtype=np.uint32).reshape(8, 8)
    v = grib.ints_to_values(x, pack)
    cube = exact.ExactCube((8, 8), 1, str(tmp_path), 4, "dxy", 3, "ints", 1)
    cube.add_step(0, "2024-05-07T21:00:00Z", v, pack, b"msg")
    hdr = exact.ExactReader.from_bytes(cube.to_bytes({})).header
    cube.close()
    rec = exact.manifest_record(hdr)["steps"][0]
    assert (rec["template"], rec["R"], rec["E"], rec["D"], rec["nbits"]) == (3, 0.5, -4, 0, 9)
    assert rec["ints_sha256"] == exact.ints_sha256(x) and rec["values_sha256"] == exact.values_sha256(v)
    assert rec["source_sha256"] == hashlib.sha256(b"msg").hexdigest()


# ---------------------------------------------------------------- G2: lossless sparse cubes

def test_sparse_cube_records_and_asserts_exactness():
    pack = {"template": 3, "R": 0.09325958788394928, "E": -4, "D": 0, "nbits": 9, "bitmap": False}
    x = np.arange(12, dtype=np.uint32).reshape(3, 4) * 37
    v = grib.ints_to_values(x, pack)
    good = sparse.SparseCube((3, 4), 1, 0.0, 1, [])
    good.add_step(0, v, pack)
    data, _ = good.to_parquet({})
    assert good.max_quant_err == 0.0 and good.exact_cells == good.total_cells == 12
    assert sparse.decode_check(data, good)
    assert grib.same_bits(sparse.dense_step(data, 0), v)
    # the same 1/16 m/s field without its packing: quantised to 0.01, recorded, and the exact cube fails
    bad = sparse.SparseCube((3, 4), 1, 0.0, 1, [])
    bad.add_step(0, v, None)
    data, _ = bad.to_parquet({})
    assert 0 < bad.max_quant_err <= 0.005 + 1e-12 and bad.exact_cells < bad.total_cells
    assert not sparse.decode_check(data, bad)
    lenient = sparse.SparseCube((3, 4), 1, 0.0, 1, [], exact=False)
    lenient.add_step(0, v, None)
    assert sparse.decode_check(lenient.to_parquet({})[0], lenient)


# ---------------------------------------------------------------- G1: MESH display encoding

def test_mesh_codes_are_specials_not_overflow():
    enc = tile.Encoding(load_config()["encodings"]["mesh05"])
    v = np.array([[-3.0, -1.0, 0.0, 0.8], [12.7, 76.2, -999.0, np.nan]])
    codes, ov = enc.encode(v)
    assert not ov.any()
    assert codes[0, 0] == 251 and codes[0, 1] == 250 and codes[1, 2] == 253 and codes[1, 3] == 254
    d = enc.decode(codes)
    assert d[0, 0] == -3.0 and d[0, 1] == -1.0
    assert np.all(np.abs(d[0, 2:] - v[0, 2:]) <= enc.max_error())
    cfg = load_config()
    mesh = next(o for g in cfg["noaa"]["groups"] for o in g["outputs"] if o.get("layer") == "mesh-u8")
    assert mesh["enc"] == "mesh05" and mesh["ver"] == "v2" and mesh["enabled"] is False
