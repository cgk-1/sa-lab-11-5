import hashlib
import struct

import numpy as np
import pytest

from feeder import compact, grib, load_config, noaa, sparse, stormbox, tile
from feeder.selftest import synthetic_csv


# ---------------------------------------------------------------- Parquet (rt)

@pytest.mark.parametrize("data", [
    synthetic_csv(),
    synthetic_csv("\r\n"),
    synthetic_csv(quoted=True),
    b"a,b\n1,2",                      # no final newline
    b"a,b\n",                         # header only
    b"#DEBUG x\n#y\na;b\n1;2\n",      # preamble (with a non-default delimiter below)
    "station,remark\nK1,café résumé\n".encode("utf-8"),
    "station,remark\nK1,caf\xe9\n".encode("latin-1"),
    b'a,b\n"x\ny",2\n',               # quoted newline
    b"a,b\n,\n,\n",                   # empty fields
])
def test_csv_round_trip_exact(data):
    delim = ";" if b";" in data else ","
    t = compact.csv_to_table(data, delim)
    out = compact.table_to_bytes(t)
    assert out == data
    p = compact.table_to_parquet(t)
    assert compact.roundtrip_ok([p], hashlib.sha256(data).hexdigest())


@pytest.mark.parametrize("data", [b"", b"a,b\n1,2,3\n", b"a,b\r\n1,2\n", b'a,b\n"1,2\n'])
def test_not_round_trippable(data):
    with pytest.raises(compact.NotRoundTrippable):
        compact.csv_to_table(data)


def test_month_reply_sliced_to_days_rejoins_exactly():
    days = [f"2024-01-{d:02d}" for d in range(1, 32)]
    data = synthetic_csv(days=days, stations=("A", "B", "C"))
    t = compact.csv_to_table(data)
    sl = compact.slice_by_day(t, ["valid"])
    assert sorted(sl) == days
    parts = [compact.table_to_parquet(x) for x in sl.values()]
    assert compact.roundtrip_ok(parts, hashlib.sha256(data).hexdigest())
    assert not compact.roundtrip_ok(parts[:-1], hashlib.sha256(data).hexdigest())
    # a day request equals the same day's slice of the month request
    day = compact.csv_to_table(synthetic_csv(days=["2024-01-07"], stations=("A", "B", "C")))
    assert compact.table_to_bytes(sl["2024-01-07"]) == compact.table_to_bytes(day)


def test_tar_and_zstd():
    members = [("a/x.grib2.gz", b"1" * 100), ("b/y.grib2.gz", b"2" * 10)]
    tb = compact.tar_bytes(members)
    assert compact.tar_bytes(members) == tb  # deterministic
    assert compact.untar(tb) == members
    assert compact.unzstd(compact.zstd(tb)) == tb


# ---------------------------------------------------------------- tiles / PMTiles

def test_tileid_known_values():
    assert tile.zxy_to_tileid(0, 0, 0) == 0
    assert [tile.zxy_to_tileid(1, x, y) for x, y in ((0, 0), (0, 1), (1, 1), (1, 0))] == [1, 2, 3, 4]
    ids = {tile.zxy_to_tileid(3, x, y) for x in range(8) for y in range(8)}
    assert ids == set(range(21, 85))


def test_encoding_specials_nan_overflow():
    enc = tile.Encoding(load_config()["encodings"]["dbz05"])
    v = np.array([[-999.0, -99.0, np.nan, -32.0], [10.25, 93.5, 200.0, -50.0]])
    codes, ov = enc.encode(v)
    assert codes[0].tolist() == [253, 252, 254, 0]
    assert ov.tolist() == [[False] * 4, [False, False, True, True]]
    d = enc.decode(codes)
    assert d[0, 0] == -999.0 and d[0, 1] == -99.0 and np.isnan(d[0, 2])
    assert abs(d[1, 0] - 10.25) <= enc.max_error()


def test_pmtiles_many_tiles_use_leaf_directories(tmp_path):
    rng = np.random.default_rng(3)

    def content(x, y):
        return struct.pack(">II", x, y) * int(1 + (x * 7 + y * 13) % 23)

    tiles = sorted((tile.zxy_to_tileid(8, x, y), content(x, y)) for x in range(256) for y in range(200)
                   if rng.random() < 0.9 or (x, y) in ((0, 0), (255, 199), (64, 17)))
    p = tmp_path / "t.pmtiles"
    st = tile.write_pmtiles(str(p), tiles, {"k": 1}, 8, 8)
    data = p.read_bytes()
    r = tile.PMTilesReader.from_bytes(data)
    assert r.leaf_len > 0 and r.root_len <= 16384 - 127 and st["addressed"] == len(tiles)
    for x, y in ((0, 0), (255, 199), (64, 17)):
        assert r.get(8, x, y) == content(x, y)
    assert r.get(8, 0, 250) is None
    assert r.metadata() == {"k": 1}


def test_tile_cube_overview_and_sidecar(tmp_path):
    enc = tile.Encoding(load_config()["encodings"]["mph05"])
    cube = tile.TileCube((10, 12), 2, enc, str(tmp_path), 8)
    v = np.full((10, 12), 5.0)
    v[0, 0] = 70.0  # 156.6 mph -> overflow
    cube.add_step(0, v)
    cube.add_step(1, v * 0.5)
    ov = cube.overview(4, str(tmp_path))
    assert ov.shape == (3, 3)
    assert ov.decoded_step(0)[0, 0] == 70.0
    p = str(tmp_path / "c.pmtiles")
    cube.write(p, {})
    assert cube.verify_file(p)
    side = compact.parquet_to_table(cube.sidecar_parquet())
    assert side.column("value").to_pylist() == [70.0]
    cube.close()
    ov.close()


# ---------------------------------------------------------------- GRIB2 header parsing and .idx

def _sections(ids=(0, 2, 22), surface=(103, 10)):
    s1 = struct.pack(">IB", 21, 1) + bytes(16)
    s4 = struct.pack(">IB", 34, 4) + struct.pack(">HH", 0, 0) + bytes([ids[1], ids[2]]) + bytes(11) + \
        bytes([surface[0], 0]) + struct.pack(">I", surface[1]) + bytes(6)
    body = s1 + s4 + b"7777"
    return b"GRIB" + bytes(2) + bytes([ids[0], 2]) + struct.pack(">Q", 16 + len(body)) + body


def test_parse_header_and_match():
    h = grib.parse_header(_sections())
    assert (h["discipline"], h["category"], h["number"], h["surface_type"], h["surface_value"]) == (0, 2, 22, 103, 10)
    assert grib.matches(h, {"grib": [0, 2, 22]})
    assert grib.matches(h, {"grib": [0, 2, 22], "walk": {"surface": [103, 10]}})
    assert not grib.matches(h, {"grib": [0, 2, 22], "walk": {"surface": [1, 0]}})
    assert not grib.matches(h, {"grib": [0, 2, 1]})
    with pytest.raises(grib.NeedMore):
        grib.parse_header(_sections()[:30])
    with pytest.raises(grib.GribError):
        grib.parse_header(b"NOPE" + bytes(20))


def test_parse_idx():
    text = "1:0:d=2024041800:GUST:10 m above ground:anl:\n2:500:d=2024041800:WIND:10 m above ground:anl:\n" \
           "3:900:d=2024041800:WDIR:10 m above ground:anl:\n"
    e = noaa.parse_idx(text)
    assert [(x.offset, x.end) for x in e] == [(0, 499), (500, 899), (900, None)]
    assert noaa.find_entry(e, "WIND:10 m above ground:anl").offset == 500
    assert noaa.find_entry(e, "GUST:surface:anl") is None
    with pytest.raises(grib.GribError):
        noaa.find_entry(e + e, "WIND:10 m above ground:anl")


def test_sparse_cube_rejects_out_of_range():
    c = sparse.SparseCube((2, 2), 1, 15.0, 1, [])  # lattice step 1: every cell kept
    with pytest.raises(ValueError):
        c.add_step(0, np.array([[-700.0, 1.0], [1.0, 1.0]]))  # below the packed-integer range


# ---------------------------------------------------------------- boxes

def test_box_rules():
    assert stormbox.box_id(0, 0) == "W125N24"
    assert len(stormbox.all_boxes()) == 30 * 13
    assert stormbox.boxes_for_point(35.0, -98.0) == [(13, 5)]
    assert sorted(stormbox.boxes_for_point(36.1, -97.1)) == [(13, 5), (13, 6), (14, 5), (14, 6)]
    lats = np.array([35.5] * 10)
    lons = np.array([-98.0] * 10)
    assert stormbox.grid_rule(np.array([60.0] * 8 + [10.0, np.nan]), lats, lons, 58.0, 8) == ["W099N34"]
    assert stormbox.grid_rule(np.array([60.0] * 7 + [10.0] * 3), lats, lons, 58.0, 8) == []
    big = [[(-110.0, 30.0), (-90.0, 30.0), (-90.0, 45.0), (-110.0, 45.0)]]
    # a box fully inside a polygon is touched although no vertex lies in it
    assert "W101N36" in stormbox.polygon_rule([big, big])
