"""Idempotent re-runs, new vintages on change, never an overwrite."""

import datetime as _dt

from feeder.daily import daily_iem
from feeder.manifest import parse
from feeder.run import lane_iem, lane_noaa

from conftest import DAY, STARTED, add_iem, add_other, add_urma, make_ctx


def _mans(s3):
    return [parse(v[0]) for k, v in sorted(s3.objects.items()) if k.startswith("man/")]


def test_second_backfill_dispatch_writes_nothing(cfg, web, clock, s3, tmp_path):
    add_urma(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"])
    lane_noaa(ctx, DAY, DAY, pilot=False)
    n_obj = len([k for k in s3.objects if k.startswith(("src/", "tiles/", "man/"))])
    later = STARTED + _dt.timedelta(hours=2)
    ctx2 = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma"], started=later, run_id="101")
    lane_noaa(ctx2, DAY, DAY, pilot=False)
    assert len([k for k in s3.objects if k.startswith(("src/", "tiles/", "man/"))]) == n_obj
    assert ctx2.store.puts == len([k for k in s3.objects if k.startswith("run/101/")])


def test_vintage_folders_differ_by_pull_time(cfg, web, clock, s3, tmp_path):
    add_iem(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-a")
    lane_iem(ctx, DAY, DAY, "pilot")
    keys1 = {k for k in s3.objects if k.startswith("src/iem-asos/")}
    assert keys1 and all("/v20240420T1207Z/" in k for k in keys1)


def test_refresh_unchanged_then_changed(cfg, web, clock, s3, tmp_path):
    add_iem(web, variant=0)
    add_other(web)
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, lane="iem", job="iem-a")
    lane_iem(ctx, DAY, DAY, "pilot")
    before = dict(s3.objects)
    # daily run 8 days later, IEM unchanged: re-pull at D+8 writes no new src object for DAY
    later = STARTED + _dt.timedelta(days=1)
    web2 = type(web)(clock)
    add_iem(web2, variant=0)
    add_other(web2)
    ctx2 = make_ctx(cfg, web2, clock, s3, tmp_path, lane="iem", job="daily-iem", started=later, run_id="200")
    target = DAY + _dt.timedelta(days=8)
    daily_iem(ctx2, target, [8], 0)
    new_for_day = [k for k in s3.objects if k not in before and f"/{DAY.isoformat()}/" in k and k.startswith("src/")]
    assert new_for_day == []
    # IEM revised the day: a new vintage, with supersedes, old objects untouched
    web3 = type(web)(clock)
    add_iem(web3, variant=7)
    add_other(web3)
    later3 = STARTED + _dt.timedelta(days=2)
    ctx3 = make_ctx(cfg, web3, clock, s3, tmp_path, lane="iem", job="daily-iem", started=later3, run_id="300")
    daily_iem(ctx3, target, [8], 0)
    new_asos = [k for k in s3.objects if k not in before and k.startswith(f"src/iem-asos/v1/2024/{DAY.isoformat()}/")]
    assert new_asos and all("/v20240422T1207Z/" in k for k in new_asos)
    for k, v in before.items():
        assert s3.objects[k] == v
    sup = [d for m in _mans(s3) if m["day"] == DAY.isoformat() for d in m["datasets"]
           if d["id"] == "iem-asos" and d["supersedes"]]
    assert sup and sup[0]["supersedes"] == "v20240420T1207Z"
