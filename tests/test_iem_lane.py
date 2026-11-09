import json
import urllib.parse

from feeder import compact
from feeder.manifest import parse, validate
from feeder.run import lane_iem, lane_verify

from conftest import DAY, add_iem, add_other, make_ctx


def _mans(s3):
    return [parse(v[0]) for k, v in sorted(s3.objects.items()) if k.startswith("man/")]


def test_pilot_iem_day(cfg, web, clock, s3, tmp_path, monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "ops@example.invalid")
    add_iem(web)
    add_other(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="pilot", job="iem-pilot")
    lane_iem(ctx, DAY, DAY, "pilot")
    mans = _mans(s3)
    for m in mans:
        assert validate(m) == []
    day = next(m for m in mans if m["day"] == DAY.isoformat())
    by = {d["id"]: d for d in day["datasets"]}
    for ds in ("iem-asos", "iem-asos1min", "iem-daily", "iem-rwis", "iem-lsr", "iem-vtec-all",
               "iem-vtec-svto-sbw", "boxflags-vtec", "spc-reports"):
        assert ds in by, ds
    asos = by["iem-asos"]
    assert asos["status"] == "complete" and asos["checks"]["roundtrip"] == "pass"
    assert {o["covers"]["network"] for o in asos["objects"]} == {"IA_ASOS", "NE_ASOS"}
    # the stored Parquet re-serialises to the reply's bytes
    for o in asos["objects"]:
        t = compact.parquet_to_table(s3.objects[o["key"]][0])
        reply = next(s for s in asos["sources"] if urllib.parse.parse_qs(urllib.parse.urlsplit(s["url"]).query)
                     ["station"][0][:2] == o["covers"]["network"][:2])
        assert compact.roundtrip_ok([s3.objects[o["key"]][0]], reply["sha256"])
        assert t.num_rows == 6
    # LSR CSV kept as served (zstd)
    lsr = by["iem-lsr"]
    assert lsr["objects"][0]["key"].endswith(".csv.zst")
    flags = json.loads(s3.objects[by["boxflags-vtec"]["objects"][0]["key"]][0])
    assert flags["boxes"] == ["W097N34", "W099N34"]
    spc = by["spc-reports"]
    # nine fixed SPC names per day; the fake serves one, the other eight 404s are listed
    assert spc["status"] == "complete" and len(spc["steps"]["missing"]) == 8
    assert spc["steps"]["expected"] == 9 and spc["ver"] == "v2"
    # NCEI listing keeps only files from the archive years
    ncei = [m for m in mans for d in m["datasets"] if d["id"] == "ncei-storm-events"]
    keys = [o["key"] for m in ncei for d in m["datasets"] if d["id"] == "ncei-storm-events" for o in d["objects"]]
    assert keys and all("d2024" in k for k in keys)


def test_iem_politeness(cfg, web, clock, s3, tmp_path, monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "ops@example.invalid")
    add_iem(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-x")
    lane_iem(ctx, DAY, DAY, "iem")
    starts = web.starts["mesonet.agron.iastate.edu"]
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert gaps and min(gaps) >= 2.0 - 1e-9
    uas = {x["headers"]["User-Agent"] for x in web.log}
    assert uas == {"public-data-mirror/1.0 (contact: ops@example.invalid)"}
    # backfill iem lane does not pull warnings (that is the vtec lane)
    assert not any("watchwarn" in u for u in web.urls())


def test_month_chunk_requests_one_per_network(cfg, web, clock, s3, tmp_path):
    import datetime as _dt

    add_iem(web)
    cfg["iem"]["datasets"] = [d for d in cfg["iem"]["datasets"] if d["id"] in ("iem-stations", "iem-asos")]
    cfg["other"]["datasets"] = []
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-m")
    start, end = _dt.date(2024, 2, 1), _dt.date(2024, 2, 29)
    lane_iem(ctx, start, end, "iem")
    asos_calls = [u for u in web.urls() if "/asos.py" in u]
    assert len(asos_calls) == 2
    mans = _mans(s3)
    days = {m["day"] for m in mans if any(d["id"] == "iem-asos" for d in m["datasets"])}
    assert len(days) == 29
    # union of the day slices rebuilds the month reply exactly
    parts = [s3.objects[k][0] for k in s3.objects if k.startswith("src/iem-asos/") and k.endswith("IA_ASOS.parquet")]
    m0 = next(m for m in mans if m["day"] == "2024-02-01")
    src = next(s for d in m0["datasets"] if d["id"] == "iem-asos" for s in d["sources"] if "IA" in s["url"])
    assert compact.roundtrip_ok(parts, src["sha256"])


def test_verify_lane_reports_and_orphans(cfg, web, clock, s3, tmp_path):
    add_iem(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-v")
    cfg2 = dict(cfg)
    cfg2["other"] = {"datasets": []}
    lane_iem(ctx, DAY, DAY, "iem")
    orphan = f"src/iem-asos/v2/2024/{DAY.isoformat()}/v20240101T0000Z/XX.parquet"
    s3.objects[orphan] = (b"x", None)
    vctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="verify", job="verify-1", run_id="900")
    assert lane_verify(vctx, [DAY], check_all=True) == 0
    rep = json.loads(next(v[0] for k, v in s3.objects.items() if k.startswith(f"vrf/{DAY.isoformat()}/900-")))
    assert rep["orphans"] == [orphan] and rep["get_checked"] == rep["objects"]
    assert orphan in s3.objects  # never deleted
    # a corrupted object is caught
    key = next(k for k in s3.objects if k.startswith("src/iem-asos/") and k.endswith("IA_ASOS.parquet"))
    data, ct = s3.objects[key]
    s3.objects[key] = (data[:-1] + bytes([data[-1] ^ 1]), ct)
    vctx2 = make_ctx(cfg, web, clock, s3, tmp_path, lane="verify", job="verify-2", run_id="901")
    assert lane_verify(vctx2, [DAY], check_all=True) > 0


def test_rwis_skips_empty_networks_and_no_results_body(cfg, web, clock, s3, tmp_path):
    from conftest import catalog_geojson

    def fn(url, p):
        if p.path == "/geojson/network/NE_RWIS.geojson":
            return 200, {}, catalog_geojson("NE_RWIS", n=0)
        if p.path == "/cgi-bin/request/rwis.py" and "IA_RWIS" in p.query:
            return 200, {}, b"Sorry, no results found for query!"
        return None

    web.route(fn)  # routes are tried in order: these before the generic fakes
    add_iem(web)
    cfg["iem"]["datasets"] = [d for d in cfg["iem"]["datasets"] if d["id"] in ("iem-stations", "iem-rwis")]
    cfg["other"]["datasets"] = []
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-r")
    lane_iem(ctx, DAY, DAY, "iem")
    rwis_calls = [u for u in web.urls() if "/rwis.py" in u]
    assert len(rwis_calls) == 1 and "IA_RWIS" in rwis_calls[0]  # NE_RWIS lists no station: never requested
    e = next(d for m in _mans(s3) for d in m["datasets"] if d["id"] == "iem-rwis")
    assert e["ver"] == "v2" and e["status"] == "complete"
    assert e["steps"] == {"expected": 1, "got": 1, "missing": []} and e["checks"]["empty"] == 1
    assert e["objects"] == []  # the "no results" text is never stored


def test_coverage_config():
    from feeder import load_config

    cfg = load_config()
    states = cfg["iem"]["states"]
    assert {"VI", "GU", "AS", "AK", "HI", "PR"} <= set(states) and len(states) == len(set(states)) == 54
    spc = next(d for d in cfg["other"]["datasets"] if d["id"] == "spc-reports")
    assert spc["ver"] == "v2" and set(spc["suffixes"]) == {
        "_rpts.csv", "_rpts_raw.csv", "_rpts_filtered.csv", "_rpts_filtered_torn.csv", "_rpts_filtered_wind.csv",
        "_rpts_filtered_hail.csv", "_rpts_raw_torn.csv", "_rpts_raw_wind.csv", "_rpts_raw_hail.csv"}
    ps = next(g for g in cfg["noaa"]["groups"] if g["id"] == "probsevere")
    assert ps["prefix"] == "ProbSevere/{ymd}/"
    import re

    assert re.search(ps["time_regex"], "ProbSevere/20240507/MRMS_PROBSEVERE_20240507_213640.json").groups() == \
        ("20240507", "213640")


def test_catalog_lane_runs_alone(cfg, web, clock, s3, tmp_path):
    from feeder.run import lane_catalog

    add_iem(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="catalog", job="catalog-x")
    assert lane_catalog(ctx) is False
    keys = sorted(k for k in s3.objects if k.startswith("src/"))
    assert keys and all(k.startswith("src/iem-stations/v2/") for k in keys)
    assert {k.rsplit("/", 1)[1].split("-")[0] for k in keys} == {"IA_ASOS", "NE_ASOS", "IA_RWIS", "NE_RWIS"}
    assert not any(u for u in web.urls() if "/cgi-bin/" in u)
