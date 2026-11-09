import base64
import copy
import hashlib
import json

import numpy as np

from feeder import compact, exact, grib, sparse, tile
from feeder.manifest import parse, validate
from feeder.run import Book, finish_day, lane_noaa, run_noaa_day

from conftest import DAY, NX, NY, URMA_PACK, add_urma, fake_decode, fake_message, make_ctx, urma_field

URMA_IDS = {"urma-gust-x", "urma-wind-x", "urma-wdir-x", "urma-gust-ex", "boxflags-urma"}


def _decoded(name, hour):
    """What the decoder gives for the fake URMA message of ``name`` at ``hour`` (the reference for bit checks)."""
    ids = {"gust": (0, 2, 22), "wind": (0, 2, 1), "wdir": (0, 2, 0)}[name]
    return fake_decode(fake_message(ids, urma_field(name, hour), URMA_PACK[name]), {"grib": None}).values


def _manifests(s3):
    return [parse(v[0]) for k, v in sorted(s3.objects.items()) if k.startswith("man/")]


def test_urma_day_objects_and_manifest_sha256(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    mans = _manifests(s3)
    assert len(mans) == 1
    m = mans[0]
    assert validate(m) == []
    assert m["schema"] == "feed-day/1" and m["day"] == DAY.isoformat()
    ids = {d["id"] for d in m["datasets"]}
    assert ids == URMA_IDS  # the u8 display tiles are off by default
    for d in m["datasets"]:
        assert d["status"] == "complete", (d["id"], d["steps"])
        assert d["vintage"] == ctx.vintage
        assert d["steps"]["expected"] == 24 and d["steps"]["got"] == 24
        assert len(d["sources"]) == 24
        for o in d["objects"]:
            stored = s3.objects[o["key"]][0]
            assert o["sha256"] == hashlib.sha256(stored).hexdigest()
            assert o["md5"] == base64.b64encode(hashlib.md5(stored).digest()).decode()
            assert o["bytes"] == len(stored)
        for s in d["sources"]:
            assert s["http"] == 206 and s["range"].startswith("bytes=")
            body = web.exact[s["url"]][2]
            a, b = s["range"][6:].split("-")
            part = body[int(a):(int(b) + 1) if b else None]
            assert hashlib.sha256(part).hexdigest() == s["sha256"]
    # manifest written last
    keys = [c[1] for c in s3.calls if c[0] == "put"]
    man_pos = max(i for i, k in enumerate(keys) if k.startswith("man/"))
    assert all(i < man_pos for i, k in enumerate(keys) if k.startswith(("src/", "tiles/")))


def test_exact_grids_are_bit_exact(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    m = _manifests(s3)[0]
    by = {d["id"]: d for d in m["datasets"]}
    for name in ("gust", "wind", "wdir"):
        e = by[f"urma-{name}-x"]
        assert e["ver"] == "v2" and e["checks"]["decode_back"] == "pass"
        assert e["checks"]["exact"] == {"steps_ints": 24, "steps_grib": 0, "steps_missing": 0}
        o = e["objects"][0]
        assert o["key"].endswith(f"urma-{name}-x-{DAY:%Y%m%d}.xgrid")
        data = s3.objects[o["key"]][0]
        r = exact.ExactReader.from_bytes(data)
        rec = o["exact"]
        assert rec["format"] == "exact-grid/2" and len(rec["steps"]) == 24
        assert data[:8] == b"XGRID002" and r.header["predictor"] == ("dxy" if name == "gust" else "med")
        for h in (0, 7, 23):
            want = _decoded(name, h)
            got = r.values(h)
            assert grib.same_bits(got, want), (name, h)  # every cell, bit for bit
            st = rec["steps"][h]
            assert (st["R"], st["E"], st["D"]) == (URMA_PACK[name]["R"], URMA_PACK[name]["E"], URMA_PACK[name]["D"])
            assert st["values_sha256"] == exact.values_sha256(want)
            assert st["ints_sha256"] == exact.ints_sha256(r.ints(h))
            src = next(s for s in e["sources"] if s["valid_utc"] == st["valid_utc"])
            assert st["source_sha256"] == src["sha256"]


def test_exact_decode_back_catches_corruption(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    key = next(k for k in s3.objects if k.startswith("src/urma-gust-x/"))
    data = bytearray(s3.objects[key][0])
    assert exact.verify(bytes(data))[0]
    data[-10] ^= 0x01  # one bit in the last tile
    ok, rep = exact.verify(bytes(data))
    assert not ok and rep["failed"] == [23]


def test_unpacked_message_is_kept_as_the_source_bytes(cfg, web, clock, s3, tmp_path):
    """A message whose packed integers cannot be recovered is stored as the original bytes (still exact)."""
    add_urma(web, packed=False)
    cfg = copy.deepcopy(cfg)
    g = next(g for g in cfg["noaa"]["groups"] if g["id"] == "urma")
    g["outputs"] = [o for o in g["outputs"] if o.get("dataset") == "urma-gust-x"]
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    e = _manifests(s3)[0]["datasets"][0]
    assert e["checks"]["exact"] == {"steps_ints": 0, "steps_grib": 24, "steps_missing": 0}
    assert e["checks"]["decode_back"] == "pass" and e["status"] == "complete"
    r = exact.ExactReader.from_bytes(s3.objects[e["objects"][0]["key"]][0])
    src = e["sources"][4]
    assert hashlib.sha256(r.source(4)).hexdigest() == src["sha256"]
    assert np.array_equal(r.values(4, fake_decode), urma_field("gust", 4))


def test_sparse_cube_exact(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    ex = next(d for d in _manifests(s3)[0]["datasets"] if d["id"] == "urma-gust-ex")
    assert ex["ver"] == "v2"
    assert ex["checks"]["max_quant_err"] == 0.0 and ex["checks"]["exact_cells"] == ex["checks"]["total_cells"] > 0
    data = s3.objects[ex["objects"][0]["key"]][0]
    for h in (0, 13):
        dense = sparse.dense_step(data, h)
        src = _decoded("gust", h)
        kept = np.isfinite(dense)
        # threshold cells (>= 35 mph) are all kept, and every kept value equals the decoded source bit for bit
        assert kept[src * sparse.MPH_PER_MS >= 35.0].all()
        assert grib.same_bits(dense[kept], src[kept])
        assert kept[::10, ::10].all()


def test_display_tiles_are_optional(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    cfg = copy.deepcopy(cfg)
    for g in cfg["noaa"]["groups"]:
        for o in g["outputs"]:
            if o["kind"] == "tile":
                assert o["enabled"] is False
                o["enabled"] = True
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    tkey = next(k for k in s3.objects if k.startswith("tiles/urma-gust-u8/") and k.endswith(".pmtiles"))
    r = tile.PMTilesReader.from_bytes(s3.objects[tkey][0])
    meta = r.metadata()
    assert meta["grid_shape"] == [NY, NX] and meta["steps"] == 24
    img = tile.png_decode(r.get(r.min_zoom, 0, 0))
    enc = tile.Encoding(cfg["encodings"]["mph05"])
    dec = enc.decode(img[3 * 128:3 * 128 + NY, :NX])
    assert np.all(np.abs(dec - urma_field("gust", 3)) <= enc.max_error())
    flags = json.loads(s3.objects[next(k for k in s3.objects if k.startswith("src/boxflags-urma/"))][0])
    assert flags["boxes"] == ["W101N30"]


def test_header_walk_when_no_idx(cfg, web, clock, s3, tmp_path):
    """Before the .idx era each message is found by walking GRIB2 section headers with range reads."""
    add_urma(web, no_idx=True)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    m = _manifests(s3)[0]
    ex = next(d for d in m["datasets"] if d["id"] == "urma-gust-ex")
    assert ex["status"] == "complete" and ex["steps"]["got"] == 24
    assert "no-idx" in ex["era"]
    gx = next(d for d in m["datasets"] if d["id"] == "urma-gust-x")
    assert gx["checks"]["decode_back"] == "pass" and gx["checks"]["exact"]["steps_ints"] == 24
    # every source is one whole GRIB2 message read by byte range
    for s in ex["sources"]:
        body = web.exact[s["url"]][2]
        a, b = (int(x) for x in s["range"][6:].split("-"))
        assert body[a:a + 4] == b"GRIB" and body[b - 3:b + 1] == b"7777"
    data = s3.objects[next(k for k in s3.objects if k.startswith("src/urma-gust-ex/"))][0]
    dense = sparse.dense_step(data, 5)
    kept = np.isfinite(dense)
    assert grib.same_bits(dense[kept], _decoded("gust", 5)[kept])


def test_missing_hours_404_count_unless_known(cfg, web, clock, s3, tmp_path):
    add_urma(web, missing_hours=(1, 2))
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    m = _manifests(s3)[0]
    ex = next(d for d in m["datasets"] if d["id"] == "urma-gust-ex")
    assert [(x["http"], x["known_gap"]) for x in ex["steps"]["missing"]] == [(404, False), (404, False)]
    assert ex["status"] == "partial"  # a 404 is not a known source gap, so the day is not complete
    assert ex["steps"]["got"] == 22
    gx = next(d for d in m["datasets"] if d["id"] == "urma-gust-x")
    assert gx["checks"]["exact"]["steps_missing"] == 2
    steps = gx["objects"][0]["exact"]["steps"]
    assert [s["mode"] for s in steps[:3]] == ["ints", "missing", "missing"]


def test_known_gaps_are_complete(cfg, web, clock, s3, tmp_path):
    add_urma(web, missing_hours=(1, 2))
    cfg = copy.deepcopy(cfg)
    g = next(g for g in cfg["noaa"]["groups"] if g["id"] == "urma")
    g["known_gaps"] = [f"{DAY.isoformat()}T01:00:00Z", f"{DAY.isoformat()}T02:00:00Z"]
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    ex = next(d for d in _manifests(s3)[0]["datasets"] if d["id"] == "urma-gust-ex")
    assert [x["known_gap"] for x in ex["steps"]["missing"]] == [True, True] and ex["status"] == "complete"


def test_hrrr_has_gap_rule_and_known_gaps(cfg):
    g = next(g for g in cfg["noaa"]["groups"] if g["id"] == "hrrr")
    assert g["gap_rule"] is True
    assert "2023-06-08T17:00:00Z" in g["known_gaps"] and "2023-08-15T23:00:00Z" in g["known_gaps"]


def test_gap_rule_counts_404_runs(cfg, web, clock, s3, tmp_path):
    add_urma(web, missing_hours=tuple(range(5)))
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    assert DAY in book.failed_days  # five unlisted 404 hours: more than gap_unexplained_max


def test_gap_rule_fails_day_on_unexplained_missing(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    for h in range(6):
        url = f"https://noaa-urma-pds.s3.amazonaws.com/urma2p5.{DAY:%Y%m%d}/urma2p5.t{h:02d}z.2dvaranl_ndfd.grb2_wexp"
        web.exact[url] = (503, {}, b"")
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    assert DAY in book.failed_days
    ex = book.entries[(DAY, "urma-gust-ex", "v2")]
    assert [m["http"] for m in ex["steps"]["missing"]] == [503] * 6
    assert all(m["tries"] == 3 for m in ex["steps"]["missing"])
    failed = finish_day(ctx, book, DAY, pilot=False)
    assert ex["status"] == "partial"
    assert failed and ctx.ledger.totals["days_failed"] == 1


def test_mesh_and_probsevere_bundles(cfg, web, clock, s3, tmp_path):
    import gzip

    from conftest import fake_message

    def fn(url, p):
        if p.netloc != "noaa-mrms-pds.s3.amazonaws.com":
            return None
        if "MESH_Max" in p.path:
            v = np.zeros((NY, NX))
            v[0:6, 0:6] = 30.0
            return 200, {}, gzip.compress(fake_message((209, 3, 30), v), mtime=0)
        if p.path == "/" and "ProbSevere" in p.query:
            assert f"prefix=ProbSevere%2F{DAY:%Y%m%d}%2F" in p.query, p.query
            keys = "".join(f"<Contents><Key>ProbSevere/{DAY:%Y%m%d}/MRMS_PROBSEVERE_{DAY:%Y%m%d}_"
                           f"{h:02d}0040.json</Key><ETag>\"e{h}\"</ETag><Size>10</Size></Contents>" for h in range(3))
            return 200, {}, ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">' + keys +
                             "<IsTruncated>false</IsTruncated></ListBucketResult>").encode()
        if p.path.endswith(".json"):
            return 200, {}, b'{"features": []}'
        return None

    web.route(fn)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["mrms-mesh", "probsevere"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    m = _manifests(s3)[0]
    by = {d["id"]: d for d in m["datasets"]}
    assert "mesh-u8" not in by  # display tiles off by default
    tar = by["mrms-mesh-raw"]
    assert tar["status"] == "complete" and tar["steps"]["got"] == 48 and tar["checks"]["roundtrip"] == "pass"
    members = compact.untar(s3.objects[tar["objects"][0]["key"]][0])
    assert len(members) == 48 and compact.member_records(members) == tar["checks"]["members"]
    assert by["boxflags-mesh"]["status"] == "complete"
    ps = by["probsevere-json"]
    assert ps["ver"] == "v2" and ps["status"] == "complete"
    assert ps["checks"]["members"] == 3 and ps["objects"][0]["key"].endswith(".tar.zst")
    assert len(compact.untar(compact.unzstd(s3.objects[ps["objects"][0]["key"]][0]))) == 3
    # NODD concurrency cap: never more than 8 workers
    from feeder import noaa
    assert noaa.NODD_WORKERS == 8


def test_empty_probsevere_listing_is_missing(cfg, web, clock, s3, tmp_path):
    def fn(url, p):
        if p.netloc == "noaa-mrms-pds.s3.amazonaws.com" and p.path == "/":
            return 200, {}, (b'<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">'
                             b"<IsTruncated>false</IsTruncated></ListBucketResult>")
        return None

    web.route(fn)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["probsevere"])
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    e = book.entries[(DAY, "probsevere-json", "v2")]
    assert e["status"] == "missing" and DAY in book.failed_days


def test_station_points_from_stored_catalog(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])  # seeds a 3-station catalogue
    lane_noaa(ctx, DAY, DAY, pilot=False)
    m = _manifests(s3)[0]
    ex = next(d for d in m["datasets"] if d["id"] == "urma-gust-ex")
    assert ex["checks"]["station_points"] == 3 and ex["status"] == "complete"
    gx = next(d for d in m["datasets"] if d["id"] == "urma-gust-x")
    smap = gx["station_map"]  # the group's first station output carries the map
    assert ex["station_map"] == {"same_as": "urma-gust-x"}
    assert smap["catalog"] == "iem-stations/v2"
    # conftest grid: 0.1-degree lat/lon from (30, -100); stations at (30.1k, -100 + 0.1k), k = 1..3
    assert [s[0] for s in smap["stations"]] == ["IA0", "IA1", "IA2"]
    assert [s[1] for s in smap["stations"]] == [1 * NX + 1, 2 * NX + 2, 3 * NX + 3]
    assert [s[2] for s in smap["stations"]] == [s[1] for s in smap["stations"]]
    t, meta = sparse.read_parquet(s3.objects[ex["objects"][0]["key"]][0])
    cells = t.column("cell").to_numpy()[t.column("step").to_numpy() == 0]
    flags = t.column("flags").to_numpy()[t.column("step").to_numpy() == 0]
    halo = {(1 + dj) * NX + 1 + di for dj in (-1, 0, 1) for di in (-1, 0, 1)}
    assert halo <= set(cells.tolist())  # a 3x3 halo around each station
    assert flags[cells.tolist().index(NX + 1)] & sparse.FLAG_DEGREE
    assert meta["n_degree_cells"] == 3


def test_station_map_degree_rule():
    """The squared-degree rule (min squared lon/lat degree differences) can differ from the projected nearest point."""
    from feeder.noaa import station_map

    lats, lons = np.meshgrid(np.linspace(30, 31, 11), np.linspace(-100, -99, 11), indexing="ij")
    lons = lons + lats * 0.3  # a skewed grid
    g = grib.Grid(lats.shape, lambda: (lats, lons), lambda la, lo: 5 * 11 + 5, ("t",))
    m = station_map(g, [("X", 30.52, -99.3 + 30.52 * 0.3 + 0.04)])
    d = (lats - 30.52) ** 2 + (lons - (-99.3 + 30.52 * 0.3 + 0.04)) ** 2
    assert m[0][1] == 60 and m[0][2] == int(np.argmin(d))


def test_missing_catalog_marks_station_outputs_partial(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"], catalog=False)
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    assert DAY in book.failed_days
    finish_day(ctx, book, DAY, pilot=False)
    by = {d["id"]: d for m in _manifests(s3) for d in m["datasets"]}
    assert by["urma-gust-ex"]["checks"]["station_points"] == "unavailable"
    assert by["urma-gust-ex"]["status"] == "partial" and by["urma-gust-x"]["status"] == "partial"
    assert by["urma-wind-x"]["status"] == "complete"  # no station cells wanted


def test_verify_lane_reads_exact_grid_headers(cfg, web, clock, s3, tmp_path):
    from feeder.run import lane_verify

    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    vctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="verify", job="verify-x", run_id="902")
    assert lane_verify(vctx, [DAY], check_all=True) == 0
    rep = json.loads(next(v[0] for k, v in s3.objects.items() if k.startswith(f"vrf/{DAY.isoformat()}/902-")))
    assert rep["xgrid_checked"] == 3 and rep["orphans"] == []
    key = next(k for k in s3.objects if k.startswith("src/urma-wind-x/"))
    data, ct = s3.objects[key]
    s3.objects[key] = (b"NOTXGRID" + data[8:], ct)
    vctx2 = make_ctx(cfg, web, clock, s3, tmp_path, lane="verify", job="verify-y", run_id="903")
    assert lane_verify(vctx2, [DAY], check_all=True) > 0


def test_reflectivity_is_under_the_gap_rule():
    import json, pathlib
    cfg = json.loads((pathlib.Path(__file__).parent.parent / "feeder" / "config" / "datasets.json").read_text())
    g = next(g for g in cfg["noaa"]["groups"] if g["id"] == "mrms-refl")
    assert g["gap_rule"] is True and g["known_gaps"] == []


def _classes(s3):
    return {k: s3.classes.get(k, "STANDARD") for k in s3.objects}


def test_archive_objects_go_to_infrequent_access_and_the_rest_stays_standard(cfg, web, clock, s3, tmp_path):
    from feeder import noaa

    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    cls = _classes(s3)
    xgrids = [k for k in cls if k.endswith(".xgrid")]
    assert len(xgrids) == 3 and all(cls[k] == "STANDARD_IA" for k in xgrids)
    others = [k for k in cls if not k.endswith(".xgrid") and not k.startswith("src/iem-stations/")]
    assert any(k.startswith("man/") for k in others) and any(k.startswith("run/") for k in others)
    assert any(k.endswith(".parquet") for k in others) and any(k.endswith(".json") and k.startswith("src/")
                                                               for k in others)
    assert all(cls[k] == "STANDARD" for k in others), {k: cls[k] for k in others if cls[k] != "STANDARD"}
    # the tar bundles (MESH, ProbSevere) are archive objects too; sparse cubes and box flags are not
    st = cfg["storage"]
    assert st["archive_class_on"] is True and st["archive_class"] == "STANDARD_IA"
    assert noaa.storage_class_for(cfg, {"kind": "tar"}) == "STANDARD_IA"
    assert noaa.storage_class_for(cfg, {"kind": "exact"}) == "STANDARD_IA"
    for kind in ("sparse", "boxflags", "tile"):
        assert noaa.storage_class_for(cfg, {"kind": kind}) is None


def test_archive_class_switch_off_writes_everything_standard(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    cfg = copy.deepcopy(cfg)
    cfg["storage"]["archive_class_on"] = False
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    assert set(_classes(s3).values()) == {"STANDARD"}
    assert all(d["status"] == "complete" for d in _manifests(s3)[0]["datasets"])


def test_verify_lane_passes_on_infrequent_access_objects(cfg, web, clock, s3, tmp_path):
    from feeder.run import lane_verify

    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    assert "STANDARD_IA" in set(_classes(s3).values())
    vctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="verify", job="verify-ia", run_id="904")
    assert lane_verify(vctx, [DAY], check_all=True) == 0
