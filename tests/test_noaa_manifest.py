import base64
import hashlib
import json

import numpy as np

from feeder import compact, sparse, tile
from feeder.manifest import parse, validate
from feeder.run import Book, finish_day, lane_noaa, run_noaa_day

from conftest import DAY, NX, NY, add_urma, make_ctx, urma_field


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
    assert ids == {"urma-gust-ex", "urma-gust-u8", "urma-wind-u8", "urma-wdir-u8", "boxflags-urma"}
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


def test_sparse_cube_exact_and_tiles_decode(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=True)
    ex_key = next(k for k in s3.objects if k.startswith("src/urma-gust-ex/"))
    data = s3.objects[ex_key][0]
    for h in (0, 13):
        dense = sparse.dense_step(data, h)
        src = urma_field("gust", h)
        kept = np.isfinite(dense)
        # threshold cells (>= 35 mph) are all kept, and every kept value equals the source at 0.01 m/s
        assert kept[src * sparse.MPH_PER_MS >= 35.0].all()
        assert np.array_equal(np.rint(src[kept] * 100), np.rint(dense[kept] * 100))
        assert kept[::10, ::10].all()
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
    # every source is one whole GRIB2 message read by byte range
    for s in ex["sources"]:
        body = web.exact[s["url"]][2]
        a, b = (int(x) for x in s["range"][6:].split("-"))
        assert body[a:a + 4] == b"GRIB" and body[b - 3:b + 1] == b"7777"
    data = s3.objects[next(k for k in s3.objects if k.startswith("src/urma-gust-ex/"))][0]
    dense = sparse.dense_step(data, 5)
    kept = np.isfinite(dense)
    assert np.array_equal(np.rint(urma_field("gust", 5)[kept] * 100), np.rint(dense[kept] * 100))


def test_missing_hours_404_listed_and_gap_rule(cfg, web, clock, s3, tmp_path):
    add_urma(web, missing_hours=(1, 2))
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    m = _manifests(s3)[0]
    ex = next(d for d in m["datasets"] if d["id"] == "urma-gust-ex")
    assert [x["http"] for x in ex["steps"]["missing"]] == [404, 404]
    assert ex["status"] == "complete"  # source gaps (404) are listed, not failures
    assert ex["steps"]["got"] == 22


def test_gap_rule_fails_day_on_unexplained_missing(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    for h in range(6):
        url = f"https://noaa-urma-pds.s3.amazonaws.com/urma2p5.{DAY:%Y%m%d}/urma2p5.t{h:02d}z.2dvaranl_ndfd.grb2_wexp"
        web.exact[url] = (503, {}, b"")
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    assert DAY in book.failed_days
    ex = book.entries[(DAY, "urma-gust-ex", "v1")]
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
        if p.path == "/" and "PROBSEVERE" in p.query:
            keys = "".join(f"<Contents><Key>ProbSevere/PROBSEVERE/{DAY:%Y%m%d}/MRMS_PROBSEVERE_{DAY:%Y%m%d}_"
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
    tar = by["mrms-mesh-raw"]
    assert tar["status"] == "complete" and tar["steps"]["got"] == 48 and tar["checks"]["roundtrip"] == "pass"
    members = compact.untar(s3.objects[tar["objects"][0]["key"]][0])
    assert len(members) == 48 and compact.member_records(members) == tar["checks"]["members"]
    assert by["boxflags-mesh"]["status"] == "complete"
    ps = by["probsevere-json"]
    assert ps["checks"]["members"] == 3 and ps["objects"][0]["key"].endswith(".tar.zst")
    assert len(compact.untar(compact.unzstd(s3.objects[ps["objects"][0]["key"]][0]))) == 3
    # NODD concurrency cap: never more than 8 workers
    from feeder import noaa
    assert noaa.NODD_WORKERS == 8


def test_station_points_from_stored_catalog(cfg, web, clock, s3, tmp_path):
    from conftest import catalog_geojson

    s3.objects["src/iem-stations/v1/2024/2024-04-01/v20240401T1113Z/IA_ASOS-0123456789abcdef.geojson.zst"] = (
        compact.zstd(catalog_geojson("IA_ASOS")), "application/zstd")
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    m = _manifests(s3)[0]
    ex = next(d for d in m["datasets"] if d["id"] == "urma-gust-ex")
    assert ex["checks"]["station_points"] == 3
