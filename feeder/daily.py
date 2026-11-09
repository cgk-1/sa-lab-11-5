"""Daily append: ``python -m feeder.daily``.

LANE=daily-noaa: gridded sources for day D (IN_DAY, blank = yesterday UTC); a new vintage of D-3 for any group whose
source ETags changed or whose missing steps have since appeared; verify D.
LANE=daily-iem: observation and other sources for D in day shape; re-pulls at REFRESH_OFFSETS (a new vintage only
when the reply changed); verify D plus VERIFY_SEEDED_DAYS seeded older days; write the day index.
"""

import datetime as _dt
import json
import os
import random
import sys

from . import canonical_json, iem, load_config, log, noaa, other, utcnow
from .breaker import BreakerTripped
from .manifest import complete_set, day_manifests, newest_entry
from .plan import ARCHIVE_START, PlanError, parse_day
from .run import (EXIT_TRIPPED, FATAL, Book, Ctx, _env_bool, _iem_enabled, finish_day, lane_verify,
                  run_noaa_day)
from .store import Store, idx_key

INDEX_VER = "v1"


def target_day(env, today):
    raw = (env.get("IN_DAY") or "").strip()
    if not raw:
        return today - _dt.timedelta(days=1)
    return parse_day(raw, "day", today)


def daily_noaa(ctx, day):
    book = Book(ctx)
    run_noaa_day(ctx, book, day)
    finish_day(ctx, book, day, pilot=False)
    back = day - _dt.timedelta(days=3)
    if back >= ARCHIVE_START:
        mans = day_manifests(ctx.store, back)
        changed, sup = [], {}
        for g in ctx.cfg["noaa"]["groups"]:
            entries = [newest_entry(mans, noaa.output_id(o), o["ver"]) for o in g["outputs"]]
            entries = [e for e in entries if e]
            if entries and noaa.source_etags_changed(ctx, g, back, entries):
                changed.append(g["id"])
                for e in entries:
                    sup[e["id"]] = e["vintage"]
        if changed:
            rb = Book(ctx)
            run_noaa_day(ctx, rb, back, force=True, only_groups=set(changed), supersedes_by_output=sup)
            finish_day(ctx, rb, back, pilot=False)
    return lane_verify(ctx, [day])


def daily_iem(ctx, day, offsets, seeded):
    book = Book(ctx)
    today = utcnow().date()
    catalogs = {}
    done = complete_set(day_manifests(ctx.store, day))
    for ds in _iem_enabled(ctx):
        ctx.breaker.check_before_day()
        skip = frozenset([day]) if (ds["id"], ds["ver"]) in done else frozenset()
        if ds["kind"] == "catalog":
            catalogs = iem.run_catalog(ctx, book, ds, today)
        elif ds["kind"] == "table":
            iem.run_table(ctx, book, ds, day, day, "day", catalogs, skip)
        elif ds["kind"] == "asserved":
            iem.run_asserved(ctx, book, ds, day, day, skip)
    for ds in ctx.cfg["other"]["datasets"]:
        if not ds.get("enabled"):
            continue
        if ds["kind"] == "listing":
            other.run_listing(ctx, book, ds, today)
        elif ds["kind"] == "daily-files":
            other.run_daily_files(ctx, book, ds, day, day,
                                  frozenset([day]) if (ds["id"], ds["ver"]) in done else frozenset())
    # re-pulls of past days
    for k in offsets:
        past = day - _dt.timedelta(days=k)
        if past < ARCHIVE_START:
            continue
        mans = day_manifests(ctx.store, past)
        for ds in _iem_enabled(ctx):
            if k not in ds.get("refresh", []) or ds["kind"] not in ("table", "asserved"):
                continue
            ctx.breaker.check_before_day()
            got = iem.refresh_day(ctx, book, ds, past, newest_entry(mans, ds["id"], ds["ver"]), catalogs)
            log("refresh", ds["id"], past.isoformat(), got)
        for ds in ctx.cfg["other"]["datasets"]:
            if ds.get("enabled") and ds["kind"] == "daily-files" and k in ds.get("refresh", []):
                staged = book.child()
                other.run_daily_files(ctx, staged, ds, past, past)
                old = newest_entry(mans, ds["id"], ds["ver"])
                new_sha = sorted(s["sha256"] for e in staged.entries.values() for s in e["sources"] if s["sha256"])
                old_sha = sorted(s["sha256"] for s in (old or {}).get("sources", []) if s.get("sha256"))
                if new_sha == old_sha:
                    book.unchanged += 1
                else:
                    staged.commit_into(book, supersedes=old["vintage"] if old else None)
    for d in book.days():
        finish_day(ctx, book, d, pilot=False)
    # verify D + seeded older days
    span = (day - ARCHIVE_START).days
    rng = random.Random(f"{ctx.run['id']}-{day.isoformat()}")
    older = sorted({ARCHIVE_START + _dt.timedelta(days=rng.randrange(span)) for _ in range(seeded)}) if span > 0 else []
    fails = lane_verify(ctx, [day] + older)
    write_index(ctx, sorted(set(book.days()) | {day}))
    return fails


def write_index(ctx, days):
    """Day index: for each day, the newest vintage and status of every dataset, missing-step counts, era flags and
    flagged boxes. Incremental: the newest stored index plus the days touched by this run."""
    if not ctx.store.online:
        return None
    prev = None
    keys = sorted(o["key"] for o in ctx.store.list(f"idx/{INDEX_VER}/"))
    if keys:
        data, _ = ctx.store.get(keys[-1])
        prev = json.loads(data.decode("utf-8"))
    doc = prev or {"schema": "feed-index/1", "days": {}}
    for day in days:
        mans = day_manifests(ctx.store, day)
        rec = {"datasets": {}, "missing_steps": 0, "era": [], "boxes": []}
        boxes = set()
        ids = sorted({(d["id"], d["ver"]) for m in mans for d in m["datasets"]})
        for ds_id, ver in ids:
            e = newest_entry(mans, ds_id, ver)
            if e is None:
                continue
            rec["datasets"][f"{ds_id}/{ver}"] = {"vintage": e["vintage"], "status": e["status"]}
            rec["missing_steps"] += len(e["steps"]["missing"])
            rec["era"] = sorted(set(rec["era"]) | set(e.get("era", [])))
            if ds_id.startswith("boxflags-"):
                for o in e["objects"]:
                    if o["key"].endswith(".json"):
                        bd, _ = ctx.store.get(o["key"])
                        boxes |= set(json.loads(bd.decode("utf-8")).get("boxes", []))
        rec["boxes"] = sorted(boxes)
        doc["days"][day.isoformat()] = rec
    doc["built_utc"] = utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    body = canonical_json(doc)
    if ctx.store.write:
        return ctx.store.put(idx_key(INDEX_VER, utcnow().date(), ctx.run["id"], ctx.run["attempt"]), body,
                             "application/json")
    return None


def main(env=None):
    env = env if env is not None else os.environ
    lane = env.get("LANE", "")
    if lane not in ("daily-noaa", "daily-iem"):
        print("daily: unknown LANE", file=sys.stderr)
        return 2
    today = utcnow().date()
    try:
        day = target_day(env, today)
    except PlanError as err:
        print(f"daily: {err}", file=sys.stderr)
        return 2
    cfg = load_config()
    store = Store.from_env(write=_env_bool("WRITE", True, env))
    ctx = Ctx(cfg, "noaa" if lane == "daily-noaa" else "iem", f"{lane}-{day.isoformat()}", store)
    log("start", lane, f"day={day}", f"write={store.write}", f"online={store.online}")
    code = 0
    try:
        if lane == "daily-noaa":
            code = 1 if daily_noaa(ctx, day) else 0
        else:
            offsets = [int(x) for x in (env.get("REFRESH_OFFSETS") or "3,8,36,120").split(",") if x.strip()]
            seeded = int(env.get("VERIFY_SEEDED_DAYS") or "7")
            code = 1 if daily_iem(ctx, day, offsets, seeded) else 0
    except BreakerTripped as tr:
        code = EXIT_TRIPPED if tr.origin else 0
    except FATAL as err:
        log("FATAL", type(err).__name__, str(err)[:200])
        code = EXIT_TRIPPED
    except Exception as err:  # anything else: type only, so no endpoint, bucket or row text reaches the public log
        log("FATAL", type(err).__name__)
        code = EXIT_TRIPPED
    finally:
        ctx.close()
    log("end", lane, f"exit={code}", f"puts={store.puts}", f"bytes={store.bytes_written}")
    return code


if __name__ == "__main__":
    sys.exit(main())
