"""Mesonet (MADIS LDAD) lane: parser, lossless day table, gap rule, budget, plan and daily catch-up."""

import copy
import datetime as _dt
import gzip
import hashlib
import io
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from feeder import load_config, madis, plan
from feeder.breaker import Breaker, Ledger
from feeder.daily import daily_madis
from feeder.manifest import parse, validate
from feeder.run import lane_madis, run_madis_day
from feeder.selftest import FILL32, FakeS3, synthetic_madis_nc
from feeder.store import Store

from conftest import STARTED, make_ctx

pytest.importorskip("scipy")

MDAY = _dt.date(2023, 4, 1)
HOST = "madis-data.ncep.noaa.gov"
N = 12


@pytest.fixture
def spec():
    return copy.deepcopy(load_config()["madis"])


def _bits(a):
    return a.view({4: np.uint32, 8: np.uint64}[a.dtype.itemsize]) if a.dtype.kind in "fi" else a


def _native_bits(a):
    return np.ascontiguousarray(a).astype(a.dtype.newbyteorder("="), copy=True).view(
        {4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


def _hour_gz(day, h, seed=None, **kw):
    body, _ = synthetic_madis_nc(hour=(day - _dt.date(2023, 4, 1)).days * 24 + h, n=N,
                                 seed=h if seed is None else seed, **kw)
    return gzip.compress(body, mtime=0)


def serve(web, day=MDAY, status=None, body=None, **kw):
    """Serves the 24 hourly files of ``day``; ``status`` / ``body``: {hour: status} / {hour: bytes} overrides."""
    status, body = status or {}, body or {}
    files = {h: _hour_gz(day, h, **kw) for h in range(24)}
    prefix = f"/madisPublic1/data/archive/{day:%Y/%m/%d}/LDAD/mesonet/netCDF/{day:%Y%m%d}_"

    def fn(url, p):
        if p.netloc != HOST or not p.path.startswith(prefix):
            return None
        h = int(p.path[len(prefix):len(prefix) + 2])
        if h in status:
            return status[h], {}, b"error"
        return 200, {}, body.get(h, files[h])

    web.route(fn)
    return files


def _mans(s3):
    return [parse(v[0]) for k, v in sorted(s3.objects.items()) if k.startswith("man/")]


def _entry(s3, day=MDAY):
    es = [d for m in _mans(s3) if m["day"] == day.isoformat() for d in m["datasets"] if d["id"] == "madis-mesonet"]
    return sorted(es, key=lambda e: e["vintage"])


# ---------------------------------------------------------------- parser

def test_parser_keeps_every_value_exactly(spec):
    body, wrote = synthetic_madis_nc(hour=3, n=N, seed=7)
    h = madis.parse_hour(body, spec, index=3)
    assert h.rows == N and h.absent == []
    assert set(h.cols) == set(madis.column_names(spec))  # only the configured variables (no "elevation")
    for name in madis.column_names(spec):
        got, src = h.cols[name], wrote[name]
        if src.dtype.kind == "S":
            want = np.ascontiguousarray(src).view(np.uint8).reshape(N, -1)
            assert np.array_equal(got, want), name  # every byte, NUL padding included
        else:
            assert got.dtype.isnative and np.array_equal(_bits(got), _native_bits(src)), name
    g = h.cols["windGust"]
    assert g[2] == FILL32 and g[3] == np.float32(-9999.0)  # fill value and missing_value kept as values
    assert g[4:5].view(np.uint32)[0] == 0x7FC01234  # NaN payload kept
    assert np.signbit(h.cols["windSpeed"][5])  # -0.0 kept
    assert bytes(h.cols["stationId"][0]).rstrip(b"\0") and len(bytes(h.cols["stationId"][0])) == 6
    a = h.attrs["windGust"]
    assert a["units"] == "meter/sec" and a["_FillValue"] == {"dtype": "float32", "values": [float(FILL32).hex()]}


def test_absent_variable_is_null_not_a_fill_value(spec):
    body, _ = synthetic_madis_nc(hour=0, n=N, absent=("precipAccum", "precipAccumQCR"))
    h0 = madis.parse_hour(body, spec, index=0)
    assert h0.cols["precipAccum"] is None and h0.absent == ["precipAccum", "precipAccumQCR"]
    h1 = madis.parse_hour(synthetic_madis_nc(hour=1, n=N, seed=1)[0], spec, index=1)
    t = pq.read_table(io.BytesIO(madis.to_parquet(spec, madis.build_table(spec, [h0, h1], {}))))
    hour = t.column("_hour").to_numpy()
    pa_ = t.column("precipAccum").combine_chunks()
    nulls = np.array(pa_.is_null().to_pylist())
    assert np.array_equal(nulls, hour == 0)  # null exactly for the hour whose file lacks the variable
    assert FILL32 in t.column("precipAccum").drop_null().to_numpy()  # the other hour's fill values stay values


@pytest.mark.parametrize("change", [{"types": {"windGust": "d"}}, {"absent": ("observationTime",)},
                                    {"absent": ("stationId",)}])
def test_type_change_or_missing_key_is_refused(spec, change):
    body, _ = synthetic_madis_nc(n=N, **change)
    with pytest.raises(madis.NotLossless):
        madis.parse_hour(body, spec)


# ---------------------------------------------------------------- lossless table

def test_order_runs_restore_any_record_order():
    rng = np.random.default_rng(3)
    for _ in range(50):
        n = int(rng.integers(1, 60))
        prov = rng.choice(np.array([b"A", b"BB", b"C"], "S2"), n)
        sid = rng.choice(np.array([b"x1", b"x2", b"y"], "S2"), n)
        obs = rng.integers(0, 4, n).astype(np.uint64)
        run = madis.order_runs(prov, sid, obs)
        perm = rng.permutation(n)  # any stored order
        back = madis.restore_order(prov[perm], sid[perm], obs[perm], run[perm])
        assert np.array_equal(perm[back], np.arange(n))


def test_day_table_round_trip_is_lossless_and_deterministic(spec):
    hours = [madis.parse_hour(synthetic_madis_nc(hour=h, n=N, seed=h)[0], spec, index=h) for h in range(5)]
    blob = madis.to_parquet(spec, madis.build_table(spec, hours, {"schema": madis.SCHEMA}))
    assert madis.decode_back_ok(spec, blob, hours) == (True, None)
    assert blob == madis.to_parquet(spec, madis.build_table(spec, hours, {"schema": madis.SCHEMA}))
    t = pq.read_table(io.BytesIO(blob))
    assert t.num_rows == 5 * N and t.column_names == ["_hour", "_run"] + madis.column_names(spec)
    assert str(t.schema.field("windGust").type) == "float" and str(t.schema.field("observationTime").type) == "double"
    assert str(t.schema.field("stationId").type) == "binary" and str(t.schema.field("windGustQCR").type) == "int32"
    assert json.loads(t.schema.metadata[madis.META_KEY])["schema"] == madis.SCHEMA


@pytest.mark.parametrize("tamper", ["value", "order", "row"])
def test_decode_back_catches_any_difference(spec, tamper):
    hours = [madis.parse_hour(synthetic_madis_nc(hour=h, n=N, seed=h)[0], spec, index=h) for h in range(2)]
    blob = madis.to_parquet(spec, madis.build_table(spec, hours, {}))
    h = hours[1]
    if tamper == "value":  # one bit of one value
        h.cols["temperature"] = h.cols["temperature"].copy()
        h.cols["temperature"][6:7].view(np.uint32)[0] ^= 1
    elif tamper == "order":  # two records swapped in the source
        for k, a in h.cols.items():
            a = a.copy()
            a[[6, 7]] = a[[7, 6]]
            h.cols[k] = a
    else:
        h.rows += 1
    assert madis.decode_back_ok(spec, blob, hours)[0] is False


def test_real_shape_char_widths_and_flags(spec):
    assert madis.flag_names(spec)[:2] == ["windGustDD", "windGustQCR"]
    assert set(spec["columns"]) >= {"stationId", "dataProvider", "latitude", "longitude", "observationTime",
                                    "windGust", "windSpeed", "windDir", "temperature", "altimeter", "precipAccum"}
    assert set(spec["required"]) >= set(spec["order_key"])
    assert madis.hour_url(spec, _dt.datetime(2023, 3, 1, 7, tzinfo=_dt.timezone.utc)) == \
        "https://madis-data.ncep.noaa.gov/madisPublic1/data/archive/2023/03/01/LDAD/mesonet/netCDF/20230301_0700.gz"


# ---------------------------------------------------------------- lane

def test_full_day_one_table_with_source_hashes(cfg, web, clock, s3, tmp_path):
    files = serve(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-x")
    lane_madis(ctx, MDAY, MDAY)
    (e,) = _entry(s3)
    assert all(validate(m) == [] for m in _mans(s3))
    assert e["status"] == "complete" and e["steps"] == {"expected": 24, "got": 24, "missing": []}
    assert e["checks"] == {"decode_back": "pass", "rows": 24 * N}
    assert [s["sha256"] for s in e["sources"]] == [hashlib.sha256(files[h]).hexdigest() for h in range(24)]
    (o,) = e["objects"]
    assert o["key"].startswith("src/madis-mesonet/v1/2023/2023-04-01/") and o["key"].endswith(
        "/madis-mesonet-20230401.parquet")
    assert not [k for k in s3.objects if k.endswith(".gz")]  # the hourly files are not kept
    t = pq.read_table(io.BytesIO(s3.objects[o["key"]][0]))
    meta = json.loads(t.schema.metadata[madis.META_KEY])
    assert [f["name"] for f in meta["files"]] == [f"20230401_{h:02d}00.gz" for h in range(24)]
    assert [f["sha256"] for f in meta["files"]] == [s["sha256"] for s in e["sources"]]
    assert meta["missing_hours"] == [] and t.num_rows == 24 * N
    assert len(web.urls(HOST)) == 24
    assert ctx.breaker.ledger.totals["days_failed"] == 0


def test_politeness_follows_the_host_config(cfg, web, clock, s3, tmp_path):
    pol = cfg["hosts"][HOST]
    assert 1 <= pol["max_inflight"] <= 2 and pol["min_interval_s"] >= 1.0 and pol["retries"] >= 3
    serve(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-p")
    lane_madis(ctx, MDAY, MDAY)
    starts = sorted(web.starts[HOST])
    assert len(starts) == 24 and min(b - a for a, b in zip(starts, starts[1:])) >= pol["min_interval_s"] - 1e-9
    ua = web.log[0]["headers"]["User-Agent"]
    assert ua.startswith("public-data-mirror/1.0")


def test_missing_hours_are_gaps_not_failures_and_are_retried(cfg, web, clock, s3, tmp_path):
    serve(web, status={5: 404, 6: 404})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-g")
    assert run_madis_day(ctx, MDAY) is False  # not a failed day
    (e,) = _entry(s3)
    assert e["status"] == "partial" and e["steps"]["got"] == 22 and e["checks"]["rows"] == 22 * N
    assert [(m["valid_utc"], m["http"], m["known_gap"]) for m in e["steps"]["missing"]] == \
        [("2023-04-01T05:00:00Z", 404, False), ("2023-04-01T06:00:00Z", 404, False)]
    assert ctx.breaker.ledger.totals["days_failed"] == 0
    # the hours appear later: the next run redoes the day (it is not complete) and supersedes the partial vintage
    web.fns.clear()
    serve(web)
    later = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-g2",
                     started=STARTED + _dt.timedelta(hours=2))
    assert run_madis_day(later, MDAY) is False
    old, new = _entry(s3)
    assert new["status"] == "complete" and new["supersedes"] == old["vintage"]
    # and a complete day is never fetched again
    n = len(web.urls(HOST))
    run_madis_day(later, MDAY)
    assert len(web.urls(HOST)) == n


def test_many_missing_hours_flag_the_day_as_gapped(cfg, web, clock, s3, tmp_path):
    serve(web, status={h: 404 for h in range(18, 24)})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-m")
    assert run_madis_day(ctx, MDAY) is False
    t = ctx.breaker.ledger.totals
    assert t["days_gapped"] == 1 and t["days_failed"] == 0


def test_known_gap_hour_is_complete(cfg, web, clock, s3, tmp_path):
    cfg["madis"]["known_gaps"] = ["2023-04-01T05:00:00Z"]
    serve(web, status={5: 404})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-k")
    run_madis_day(ctx, MDAY)
    assert _entry(s3)[0]["status"] == "complete"


def test_archive_without_the_day_stores_nothing(cfg, web, clock, s3, tmp_path):
    serve(web, status={h: 404 for h in range(24)})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-0")
    assert run_madis_day(ctx, MDAY) is False
    (e,) = _entry(s3)
    assert e["status"] == "missing" and e["objects"] == [] and len(e["steps"]["missing"]) == 24
    assert not [k for k in s3.objects if k.startswith("src/madis-mesonet/")]
    assert ctx.breaker.ledger.totals["days_gapped"] == 1


def test_5xx_is_retried_then_fails_the_day(cfg, web, clock, s3, tmp_path):
    serve(web, status={9: 503})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-5")
    assert run_madis_day(ctx, MDAY) is True
    (e,) = _entry(s3)
    tries = cfg["hosts"][HOST]["retries"]
    assert [(m["http"], m["tries"]) for m in e["steps"]["missing"]] == [(503, tries)]
    assert sum(1 for u in web.urls(HOST) if u.endswith("_0900.gz")) == tries
    assert ctx.breaker.ledger.totals["days_failed"] == 1 and ctx.breaker.ledger.totals.get("days_gapped", 0) == 0


def test_damaged_file_fails_the_day(cfg, web, clock, s3, tmp_path):
    serve(web, body={3: b"\x1f\x8b not really gzip"})
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-d")
    assert run_madis_day(ctx, MDAY) is True
    (e,) = _entry(s3)
    assert e["status"] == "partial" and [m["http"] for m in e["steps"]["missing"]] == [200]
    assert sum(1 for u in web.urls(HOST) if u.endswith("_0300.gz")) == madis.READ_ATTEMPTS


def test_failed_decode_back_uploads_nothing(cfg, web, clock, s3, tmp_path, monkeypatch):
    serve(web)
    monkeypatch.setattr(madis, "decode_back_ok", lambda spec, blob, hours: (False, "test"))
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-f")
    assert run_madis_day(ctx, MDAY) is True
    (e,) = _entry(s3)
    assert e["checks"]["decode_back"] == "fail" and e["objects"] == [] and e["status"] == "partial"
    assert not [k for k in s3.objects if k.startswith("src/madis-mesonet/")]


def test_lane_starts_at_the_dataset_start(cfg, web, clock, s3, tmp_path):
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-s")
    lane_madis(ctx, _dt.date(2021, 9, 29), _dt.date(2021, 9, 30))
    assert web.urls(HOST) == [] and _mans(s3) == []


def test_daily_redoes_incomplete_recent_days_only(cfg, web, clock, s3, tmp_path):
    d0, d1, d2 = MDAY + _dt.timedelta(days=2), MDAY + _dt.timedelta(days=1), MDAY
    for d in (d0, d1, d2):
        serve(web, day=d, status={4: 404} if d == d1 else None)
    first = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="madis-pre")
    lane_madis(first, d2, d1)  # d2 complete, d1 partial (one hour late)
    web.fns.clear()
    for d in (d0, d1, d2):
        serve(web, day=d)
    n = len(web.urls(HOST))
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="daily-madis",
                   started=STARTED + _dt.timedelta(hours=3))
    assert daily_madis(ctx, d0, catchup=2) is False
    fetched = {u.rsplit("/", 1)[1][:8] for u in web.urls(HOST)[n:]}
    assert fetched == {f"{d0:%Y%m%d}", f"{d1:%Y%m%d}"}  # D and the incomplete D-1; D-2 was complete
    assert _entry(s3, d1)[-1]["status"] == "complete" and _entry(s3, d0)[0]["status"] == "complete"


def test_dry_run_prints_the_manifest_and_uploads_nothing(cfg, web, clock, monkeypatch, capsys):
    from feeder import run as run_mod
    from feeder.net import Http

    serve(web)
    monkeypatch.setattr(run_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(run_mod, "Http", lambda c: Http(c, transport=web, sleep=clock.sleep, clock=clock))
    code = run_mod.main({"LANE": "madis", "CHUNK": f"{MDAY}..{MDAY}", "RUN_KIND": "madis", "WRITE": "false"})
    out = capsys.readouterr().out
    assert code == 0 and "puts=0" in out
    doc = json.loads(next(line for line in out.splitlines() if line.startswith('{"datasets"')))
    (e,) = doc["datasets"]
    assert e["id"] == "madis-mesonet" and e["status"] == "complete" and e["checks"]["decode_back"] == "pass"


# ---------------------------------------------------------------- budget and plan

def test_budget_uses_the_lane_days_and_other_lanes(spec):
    cfg = load_config()
    b = cfg["breaker"]
    first = _dt.date.fromisoformat(spec["start"])
    end = _dt.date(2021, 10, 1) + _dt.timedelta(days=b["total_days"] - 1)  # the day the other lanes count to
    assert b["total_days_by_lane"]["madis"] == (end - first).days + 1
    br = Breaker(b, cfg["cuts"], "madis", Ledger(Store(FakeS3(), "b", write=True), "r", "j"))
    per_day = 52e6  # the planned upper bound per day
    for _ in range(3):
        br.after_day(MDAY, per_day, False)
    proj = br.projection_gb(br.ledger.run_totals())
    assert abs(proj - (per_day * b["total_days_by_lane"]["madis"] / 1e9 + b["fixed_gb_by_lane"]["madis"])) < 1e-6
    assert proj < b["cut_gb"]
    # the mesonet footprint is counted in every other lane's fixed tier
    f = b["fixed_gb_by_lane"]
    own = (f["iem"] + f["vtec"] + f["noaa"] - 2 * f["madis"]) / 3  # fixed[lane] = total - own[lane]
    assert abs(own - per_day * b["total_days_by_lane"]["madis"] / 1e9) < 1.0


def test_plan_madis_months_from_its_start():
    today = _dt.date(2026, 11, 10)
    out = plan.plan({"IN_LANE": "madis", "IN_START": "2023-03-01", "IN_END": "2023-05-02"}, today)
    assert out["madis"] == ["2023-03-01..2023-03-31", "2023-04-01..2023-04-30", "2023-05-01..2023-05-02"]
    assert out["noaa"] == out["iem"] == out["verify"] == []
    whole = plan.plan({"IN_LANE": "madis", "IN_START": "2023-03-01", "IN_END": "2026-11-09"}, today)
    assert len(whole["madis"]) == 45
    full = plan.plan({"IN_LANE": "madis", "IN_START": "2021-10-01", "IN_END": "2023-02-28"}, today)
    assert len(full["madis"]) == 17 and full["madis"][0] == "2021-10-01..2021-10-31"
    with pytest.raises(plan.PlanError):
        plan.plan({"IN_LANE": "madis", "IN_START": "2021-09-30", "IN_END": "2021-10-02"}, today)
    assert plan.MADIS_START.isoformat() == load_config()["madis"]["start"]


def test_next_daily_index_lists_the_days_table(cfg, web, clock, s3, tmp_path):
    from feeder.daily import daily_iem

    serve(web)
    lane_madis(make_ctx(cfg, web, clock, s3, tmp_path, lane="madis", job="daily-madis"), MDAY, MDAY)
    for d in cfg["iem"]["datasets"] + cfg["other"]["datasets"]:
        d["enabled"] = False
    ictx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="daily-iem", started=STARTED + _dt.timedelta(hours=1))
    daily_iem(ictx, MDAY + _dt.timedelta(days=1), [], 0)
    key = sorted(k for k in s3.objects if k.startswith("idx/"))[-1]
    idx = json.loads(s3.objects[key][0])
    assert idx["days"][MDAY.isoformat()]["datasets"]["madis-mesonet/v1"]["status"] == "complete"
