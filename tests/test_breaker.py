import copy
import datetime as _dt

import pytest

from feeder import load_config
from feeder.breaker import Breaker, BreakerTripped, Ledger
from feeder.net import Http
from feeder.selftest import FakeS3
from feeder.store import Store

DAY = _dt.date(2024, 1, 1)


@pytest.fixture
def bcfg():
    c = copy.deepcopy(load_config())
    return c["breaker"], c["cuts"]


def _b(bcfg, lane="noaa", store=None, job="j1", sleep=None):
    store = store or Store(FakeS3(), "b", write=True)
    return Breaker(bcfg[0], bcfg[1], lane, Ledger(store, "run1", job), sleep=sleep), store


def test_integrity_share_trips_run(bcfg):
    b, _ = _b(bcfg)
    b.after_day(DAY, 10, False, verified=1000, verify_failed=5)
    b.check_before_day()  # 0.5 % is the limit, not above it
    b.after_day(DAY, 10, False, verified=0, verify_failed=1)
    with pytest.raises(BreakerTripped) as e:
        b.check_before_day()
    assert "integrity" in e.value.reason and e.value.origin


def test_day_failure_share_trips(bcfg):
    b, _ = _b(bcfg)
    for _ in range(100):
        b.after_day(DAY, 10, False)
    b.check_before_day()
    b.after_day(DAY, 10, True)
    b.after_day(DAY, 10, True)
    with pytest.raises(BreakerTripped):
        b.check_before_day()


def test_tripped_run_stops_other_jobs_cleanly(bcfg):
    b1, store = _b(bcfg)
    b1.after_day(DAY, 10, True)
    with pytest.raises(BreakerTripped):
        b1.check_before_day()
    b2, _ = _b(bcfg, store=store, job="j2")
    with pytest.raises(BreakerTripped) as e:
        b2.check_before_day()
    assert not e.value.origin  # a job that did not trip exits clean


def test_ledger_is_append_only(bcfg):
    b, store = _b(bcfg)
    for _ in range(3):
        b.after_day(DAY, 1, False)
    keys = sorted(k for k in store._client.objects if k.startswith("run/"))
    assert keys == [f"run/run1/j1/{i:06d}.json" for i in (1, 2, 3)]


def test_size_projection_applies_cuts_then_stops(bcfg):
    cfg, cuts = bcfg
    b, _ = _b(bcfg)
    per_day = (cfg["cut_gb"] + 10 - cfg["fixed_gb_by_lane"]["noaa"]) * 1e9 / cfg["total_days"]
    for _ in range(3):
        b.after_day(DAY, per_day, False)
    b.check_before_day()
    assert [c["id"] for c in b.active_cuts()] == ["refl-15min"]
    b.check_before_day()  # needs more days before the next cut
    assert len(b.active_cuts()) == 1
    enabled = [c for c in cuts if c.get("enabled")]
    huge = (cfg["stop_gb"] + 50 - cfg["fixed_gb_by_lane"]["noaa"]) * 1e9 / cfg["total_days"]
    with pytest.raises(BreakerTripped) as e:
        for _ in range(40):
            b.after_day(DAY, huge, False)
            b.check_before_day()
    assert len(b.active_cuts()) == len(enabled) and "size" in e.value.reason


def test_bytes_per_day_streak_stops_job(bcfg):
    cfg, cuts = bcfg
    cfg = dict(cfg, expected_bytes_per_day={"noaa": 1000})
    b, _ = _b((cfg, cuts))
    b.after_day(DAY, 1400, False)
    b.after_day(DAY, 1400, False)
    b.after_day(DAY, 900, False)  # streak broken
    b.after_day(DAY, 1400, False)
    b.after_day(DAY, 1400, False)
    with pytest.raises(BreakerTripped) as e:
        b.after_day(DAY, 1400, False)
    assert e.value.scope == "job"


def test_gap_rule(bcfg):
    b, _ = _b(bcfg)
    assert not b.gap_failed([{"http": 404}] * 10)
    assert not b.gap_failed([{"http": 503}] * 4)
    assert b.gap_failed([{"http": 503}] * 5)


def test_observation_lane_pause_and_stop(bcfg):
    sleeps = []
    b, _ = _b(bcfg, lane="iem", sleep=sleeps.append)
    now = [0.0]
    http = Http({"hosts": {}}, transport=lambda *a: (200, {}, b""), clock=lambda: now[0], sleep=lambda s: None)
    http.stats["bursts"] = [1.0, 2.0, 3.0]
    b.observe_http(http)
    assert sleeps == [3600] and http.stats["bursts"] == []
    http.stats.update(requests=100, failed=6)
    with pytest.raises(BreakerTripped) as e:
        b.observe_http(http)
    assert e.value.scope == "lane"


def test_pilot_has_no_size_rule(bcfg):
    cfg, _ = bcfg
    b, _ = _b(bcfg, lane="pilot")
    for _ in range(5):
        b.after_day(DAY, 10 ** 12, False)
    b.check_before_day()
    assert b.active_cuts() == []
