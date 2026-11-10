import datetime as _dt
import json

import pytest

from feeder import daily as daily_mod
from feeder import plan
from feeder import run as run_mod
from feeder.net import Http, HttpError

from conftest import DAY, add_iem, add_urma, fake_decode


# ---------------------------------------------------------------- dry run

def test_dry_run_main_prints_manifest_and_uploads_nothing(monkeypatch, capsys, web, clock, cfg):
    add_urma(web)
    add_iem(web)  # a dry run without storage reads the station catalogue live
    for n in ("S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET"):
        monkeypatch.delenv(n, raising=False)
    small = dict(cfg)
    small["noaa"] = {"groups": [g for g in cfg["noaa"]["groups"] if g["id"] == "urma"]}
    monkeypatch.setattr(run_mod, "load_config", lambda: small)
    monkeypatch.setattr(run_mod, "Http", lambda c: Http(c, transport=web, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(run_mod.grib, "decode", fake_decode)
    code = run_mod.main({"LANE": "noaa", "CHUNK": f"{DAY}..{DAY}", "RUN_KIND": "noaa", "WRITE": "false"})
    out = capsys.readouterr().out
    assert code == 0
    docs = [json.loads(line) for line in out.splitlines() if line.startswith('{"datasets"')]
    assert len(docs) == 1 and docs[0]["schema"] == "feed-day/1"
    for d in docs[0]["datasets"]:
        assert d["objects"] and all(len(o["sha256"]) == 64 for o in d["objects"])
        assert d["status"] == "complete", d["id"]
    ex = next(d for d in docs[0]["datasets"] if d["id"] == "urma-gust-ex")
    assert ex["checks"]["station_points"] == 6  # IA + NE catalogues, 3 stations each
    assert "puts=0" in out


def test_writing_run_refuses_without_credentials(monkeypatch):
    for n in ("S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET"):
        monkeypatch.delenv(n, raising=False)
    with pytest.raises(Exception):
        run_mod.main({"LANE": "noaa", "CHUNK": f"{DAY}..{DAY}", "WRITE": "true"})


def test_run_rejects_bad_inputs():
    assert run_mod.main({"LANE": "noaa", "CHUNK": "x"}) == 2
    assert run_mod.main({"LANE": "rm -rf", "CHUNK": f"{DAY}..{DAY}"}) == 2
    assert daily_mod.main({"LANE": "daily-x"}) == 2
    assert daily_mod.main({"LANE": "daily-noaa", "IN_DAY": "2020-01-01"}) == 2


# ---------------------------------------------------------------- plan

TODAY = _dt.date(2026, 10, 29)


def test_plan_noaa_weeks():
    out = plan.plan({"IN_LANE": "noaa", "IN_START": "2024-04-01", "IN_END": "2024-04-20", "IN_DPJ": "7",
                     "IN_PAR": "14"}, TODAY)
    assert out["noaa"] == ["2024-04-01..2024-04-07", "2024-04-08..2024-04-14", "2024-04-15..2024-04-20"]
    assert out["iem"] == [] and out["verify"] == [] and out["parallel"] == 14


def test_plan_iem_months_and_pilot():
    out = plan.plan({"IN_LANE": "iem", "IN_START": "2021-10-01", "IN_END": "2021-12-05"}, TODAY)
    assert out["iem"] == ["2021-10-01..2021-10-31", "2021-11-01..2021-11-30", "2021-12-01..2021-12-05"]
    p = plan.plan({"IN_LANE": "pilot", "IN_DAYS": "2023-07-01, 2022-04-05"}, TODAY)
    assert p["noaa"] == ["2022-04-05..2022-04-05", "2023-07-01..2023-07-01"] and p["parallel"] == 3
    assert p["iem"] == p["noaa"]


@pytest.mark.parametrize("env", [
    {"IN_LANE": "noaa; echo x", "IN_START": "2024-01-01", "IN_END": "2024-01-02"},
    {"IN_LANE": "noaa", "IN_START": "2024-1-1", "IN_END": "2024-01-02"},
    {"IN_LANE": "noaa", "IN_START": "2021-09-30", "IN_END": "2024-01-02"},
    {"IN_LANE": "noaa", "IN_START": "2024-01-02", "IN_END": "2026-10-29"},
    {"IN_LANE": "noaa", "IN_START": "2024-01-03", "IN_END": "2024-01-02"},
    {"IN_LANE": "noaa", "IN_START": "2024-02-30", "IN_END": "2024-03-02"},
    {"IN_LANE": "noaa", "IN_START": "2024-01-01", "IN_END": "2024-01-02", "IN_PAR": "99"},
    {"IN_LANE": "noaa", "IN_START": "2024-01-01", "IN_END": "2024-01-02", "IN_DPJ": "3"},
    {"IN_LANE": "noaa", "IN_START": "2021-10-01", "IN_END": "2026-10-28", "IN_DPJ": "7"},
    {"IN_LANE": "pilot", "IN_DAYS": ""},
    {"IN_LANE": "pilot", "IN_DAYS": ",".join(["2023-07-01"] * 11)},
    {"IN_LANE": "pilot", "IN_DAYS": "2023-07-01,$(id)"},
])
def test_plan_rejects(env):
    with pytest.raises(plan.PlanError):
        plan.plan(env, TODAY)


def test_plan_verify_fits_matrix():
    out = plan.plan({"IN_LANE": "verify", "IN_START": "2021-10-01", "IN_END": "2026-10-28"}, TODAY)
    assert len(out["verify"]) <= 256


def test_plan_main_output(monkeypatch, capsys):
    monkeypatch.setattr(plan, "today_utc", lambda: TODAY)
    for k, v in {"IN_LANE": "vtec", "IN_START": "2024-01-01", "IN_END": "2024-02-10"}.items():
        monkeypatch.setenv(k, v)
    assert plan.main() == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines == ['noaa=[]', 'iem=["2024-01-01..2024-01-31","2024-02-01..2024-02-10"]', 'verify=[]', 'madis=[]',
                     'parallel=1']


# ---------------------------------------------------------------- net

def _http(transport, clock, hosts=None, wall=None):
    cfg = {"hosts": hosts or {"*": {"max_inflight": 1, "min_interval_s": 2.0, "retries": 3, "backoff_start_s": 60,
                                    "backoff_cap_s": 1800}},
           "path_min_interval_s": {"/slow": 5.0}, "user_agent": "public-data-mirror/1.0"}
    return Http(cfg, transport=transport, sleep=clock.sleep, clock=clock, wallclock=wall)


def test_backoff_on_503_then_success(clock, monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "someone@example.invalid")
    seen = []

    def t(method, url, headers, timeout):
        seen.append(headers)
        return (503, {}, b"") if len(seen) < 3 else (200, {}, b"ok")

    h = _http(t, clock)
    r = h.get("https://example.invalid/a")
    assert r.ok and r.tries == 3 and r.body == b"ok"
    assert [s for s in clock.sleeps if s >= 60] == [60, 120]
    assert seen[0]["User-Agent"] == "public-data-mirror/1.0 (contact: someone@example.invalid)"
    assert len(h.stats["bursts"]) == 2


def test_gives_up_after_retries_and_returns_404(clock):
    h = _http(lambda *a: (503, {}, b""), clock)
    with pytest.raises(HttpError) as e:
        h.get("https://example.invalid/a")
    assert e.value.tries == 3
    h2 = _http(lambda *a: (404, {}, b""), clock)
    r = h2.get("https://example.invalid/b")
    assert r.status == 404 and r.tries == 1 and r.sha256 is None


def test_spacing_per_host_and_path(clock):
    h = _http(lambda *a: (200, {}, b"x"), clock)
    starts = []
    orig = h.transport

    def t(*a):
        starts.append(clock())
        return orig(*a)

    h.transport = t
    for _ in range(3):
        h.get("https://example.invalid/a")
    assert [round(b - a, 6) for a, b in zip(starts, starts[1:])] == [2.0, 2.0]
    starts.clear()
    for _ in range(2):
        h.get("https://example.invalid/slow")
    assert round(starts[1] - starts[0], 6) == 5.0


def test_range_header_and_server_ignoring_range(clock):
    got = {}

    def t(method, url, headers, timeout):
        got.update(headers)
        return 200, {}, b"0123456789"

    h = _http(t, clock)
    r = h.get("https://example.invalid/f", rng=(2, 4))
    assert got["Range"] == "bytes=2-4" and r.body == b"234" and r.range == "bytes=2-4"


def test_quiet_window_waits(clock):
    wall = lambda: _dt.datetime(2024, 1, 1, 9, 30, tzinfo=_dt.timezone.utc)  # noqa: E731
    hosts = {"*": {"max_inflight": 1, "min_interval_s": 0, "retries": 1, "quiet_utc": ["09:00", "10:30"]}}
    h = _http(lambda *a: (200, {}, b""), clock, hosts=hosts, wall=wall)
    h.get("https://example.invalid/a")
    assert 3600 in clock.sleeps


def test_https_only(clock):
    with pytest.raises(ValueError):
        _http(lambda *a: (200, {}, b""), clock).get("http://example.invalid/a")


def test_one_request_in_flight_per_host(monkeypatch):
    import threading
    import time as _time

    active, peak = [0], [0]
    lock = threading.Lock()

    def t(*a):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        _time.sleep(0.01)
        with lock:
            active[0] -= 1
        return 200, {}, b""

    h = Http({"hosts": {"*": {"max_inflight": 1, "min_interval_s": 0}}}, transport=t)
    ths = [threading.Thread(target=h.get, args=("https://example.invalid/x",)) for _ in range(6)]
    [x.start() for x in ths]
    [x.join() for x in ths]
    assert peak[0] == 1
