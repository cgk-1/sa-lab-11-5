"""Lane entry point for manual runs: ``python -m feeder.run``.

Environment: LANE (catalog | noaa | iem | vtec | verify; catalog needs no CHUNK), CHUNK (YYYY-MM-DD..YYYY-MM-DD), RUN_KIND (the dispatched lane:
pilot | vtec | noaa | iem | verify), WRITE (true / false; false = dry run: fetch, build and check, upload nothing,
print the manifests), CONTACT_EMAIL, FEEDER_PROCS (worker processes for the gridded lane; see feeder.par), and the
storage secrets S3_ENDPOINT / S3_BUCKET / S3_KEY_ID / S3_SECRET.
"""

import concurrent.futures as cf
import datetime as _dt
import hashlib
import os
import shutil
import sys
import tempfile
import time

from . import canonical_json, grib, iem, load_config, log, noaa, other, par, utcnow, verify
from .breaker import Breaker, BreakerTripped, Ledger
from .manifest import DatasetEntry, DayManifest, complete_set, day_manifests, iso
from .net import Http
from .plan import parse_chunk
from .store import IntegrityError, OverwriteRefused, PrefixError, Store, man_key, vintage_of, vrf_key

FATAL = (PrefixError, OverwriteRefused, IntegrityError)
EXIT_TRIPPED = 3


def _env_bool(name, default=False, env=None):
    v = (os.environ if env is None else env).get(name, "")
    if v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes")


def lock_sha256():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "requirements.lock")
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return None


def run_info(lane, job, started):
    img = "-".join(x for x in (os.environ.get("ImageOS"), os.environ.get("ImageVersion")) if x) or "local"
    return {"id": os.environ.get("GITHUB_RUN_ID", "local"), "attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
            "job": job, "lane": lane, "git_sha": os.environ.get("GITHUB_SHA", "unknown"),
            "lock_sha256": lock_sha256(), "runner_image": img, "started_utc": iso(started), "ended_utc": None}


class Ctx:
    def __init__(self, cfg, lane, job, store, http=None, decode=None, budget_min=None, sleep=time.sleep,
                 started=None, staging=None):
        self.cfg = cfg
        self.lane = lane
        self.job = job
        self.store = store
        self.write = store.write
        self.http = http or Http(cfg)
        self.decode = decode or grib.decode
        self.started = started or utcnow()
        self.vintage = vintage_of(self.started)
        self.run = run_info(lane, job, self.started)
        self.ledger = Ledger(store, self.run["id"], f"{job}-{self.run['attempt']}")
        self.breaker = Breaker(cfg["breaker"], cfg.get("cuts", []), lane, self.ledger, sleep=sleep)
        self._own_staging = staging is None
        self.staging = staging or tempfile.mkdtemp(prefix="feeder-")
        budget = budget_min if budget_min is not None else float(os.environ.get("JOB_BUDGET_MIN", "320"))
        self._deadline = time.monotonic() + budget * 60
        self._points = "unset"

    def within_budget(self):
        return time.monotonic() < self._deadline

    def station_catalog_id(self):
        ds = iem.catalog_dataset(self.cfg)
        return f"{ds['id']}/{ds['ver']}"

    def station_points(self):
        """[(station id, lat, lon)] from the stored catalogue (read live in a dry run without storage), or None."""
        if self._points == "unset":
            ds = iem.catalog_dataset(self.cfg)
            try:
                if self.store.online:
                    self._points = iem.station_points_from_store(self.store, f"src/{ds['id']}/{ds['ver']}/")
                else:
                    self._points = iem.station_points_live(self)
            except Exception as err:
                log("station-points", "unavailable", type(err).__name__)
                self._points = None
            if not self._points:
                self._points = None
                log("station-points", "unavailable", "no catalogue")
        return self._points

    def close(self):
        par.shutdown()
        if self._own_staging:
            shutil.rmtree(self.staging, ignore_errors=True)


class Book:
    """Collects dataset entries and stored objects per day; writes one manifest per day at the end."""

    def __init__(self, ctx, deferred=False):
        self.ctx = ctx
        self.deferred = deferred
        self.entries = {}
        self.written = {}
        self.bytes_by_day = {}
        self.failed_days = set()
        self.unchanged = 0
        self._pending = []

    def entry(self, day, ds_id, ver):
        k = (day, ds_id, ver)
        if k not in self.entries:
            self.entries[k] = DatasetEntry(ds_id, ver, self.ctx.vintage)
        return self.entries[k]

    def put(self, day, entry, key, data, ctype, covers=None):
        if self.deferred:
            self._pending.append((day, entry["id"], entry["ver"], key, data, ctype, covers))
            return None
        pr = self.ctx.store.put(key, data, ctype)
        entry.add_object(pr, covers)
        self.written.setdefault(day, []).append(pr)
        self.bytes_by_day[day] = self.bytes_by_day.get(day, 0) + pr["bytes"]
        return pr

    def add_group_result(self, day, res):
        for e in res.entries:
            self.entries[(day, e["id"], e["ver"])] = e
        self.written.setdefault(day, []).extend(pr for pr, _ in res.written)
        self.bytes_by_day[day] = self.bytes_by_day.get(day, 0) + res.bytes
        if res.failed:
            self.failed_days.add(day)

    # deferred staging (refresh): compare first, write only if changed
    def child(self):
        return Book(self.ctx, deferred=True)

    def pending_covers(self):
        return {c.get("network"): c.get("day_sha256") for *_, c in self._pending if c}

    def commit_into(self, parent, supersedes):
        for (day, ds_id, ver), e in self.entries.items():
            pe = parent.entry(day, ds_id, ver)
            pe["sources"].extend(e["sources"])
            for k in ("expected", "got"):
                pe["steps"][k] += e["steps"][k]
            pe["steps"]["missing"].extend(e["steps"]["missing"])
            pe["checks"].update(e["checks"])
            pe["supersedes"] = supersedes
        for day, ds_id, ver, key, data, ctype, covers in self._pending:
            parent.put(day, parent.entry(day, ds_id, ver), key, data, ctype, covers)

    def days(self):
        return sorted({d for d, _, _ in self.entries})

    def write_manifest(self, day):
        """Writes the day's manifest last (after its objects). Returns (manifest put result or None, failed)."""
        man = DayManifest(day, self.ctx.run)
        failed = day in self.failed_days
        for (d, _, _), e in sorted(self.entries.items(), key=lambda kv: (kv[0][1], kv[0][2])):
            if d != day:
                continue
            e.finish_status()
            if e["status"] == "unchanged" and not e["objects"]:
                continue
            if e["checks"].get("roundtrip") == "fail" or e["checks"].get("decode_back") == "fail":
                failed = True
            man.add(e)
        if not man.datasets:
            return None, failed
        data = man.to_bytes(ended_utc=iso(utcnow()))
        if not self.ctx.write:
            print(data.decode("ascii"), flush=True)
        pr = self.ctx.store.put(man_key(day, self.ctx.run["id"], self.ctx.run["attempt"], self.ctx.job), data,
                                "application/json")
        return pr, failed


def verify_own(ctx, objs, pilot):
    vcfg = ctx.cfg.get("verify", {})
    rep = verify.verify_objects(ctx.store, objs, vcfg.get("sample_share", 0.02), vcfg.get("min_per_day", 3),
                                seed=ctx.run["id"], check_all=pilot)
    return verify.checked(rep) if ctx.store.online else 0, verify.failures(rep)


def finish_day(ctx, book, day, pilot):
    pr, failed = book.write_manifest(day)
    objs = [o for o in book.written.get(day, []) if o["status"] in ("written", "exists-identical")]
    if pr is not None and pr["status"] in ("written", "exists-identical"):
        objs.append(pr)
    n_checked, n_failed = verify_own(ctx, objs, pilot) if ctx.write else (0, 0)
    if n_failed:
        failed = True
    ctx.breaker.after_day(day, book.bytes_by_day.get(day, 0), failed, n_checked, n_failed, book.unchanged)
    book.unchanged = 0
    log("day", day.isoformat(), f"bytes={book.bytes_by_day.get(day, 0)}", f"failed={failed}",
        f"verified={n_checked}", f"verify_failed={n_failed}")
    return failed


# ---------------------------------------------------------------- lanes

def noaa_groups(ctx):
    cuts = ctx.breaker.active_cuts()
    return [noaa.apply_cuts(g, cuts) for g in ctx.cfg["noaa"]["groups"]]


def _run_group(ctx, g, day, todo, sup):
    """One group for one day, with the configured attempts; a group that keeps failing gives a failed result."""
    attempts = int(ctx.cfg["breaker"].get("day_attempts", 3))
    last = None
    for k in range(attempts):
        try:
            return noaa.process_group(ctx, g, day, todo, supersedes=sup)
        except FATAL:
            raise
        except BreakerTripped:
            raise
        except Exception as err:
            last = type(err).__name__
            log("group-error", g["id"], day.isoformat(), last, f"attempt={k + 1}")
    res = noaa.GroupResult()
    res.failed = True
    res.reasons.append(f"error: {last}")
    return res


def run_noaa_day(ctx, book, day, force=False, only_groups=None, supersedes_by_output=None):
    """Every group for ``day``. A group marked ``overlap`` (single-threaded work that uses no worker processes, such
    as compressing one large bundle) runs on a thread beside the others; results are added in config order."""
    done = set() if force else complete_set(day_manifests(ctx.store, day))
    jobs = []
    for g in noaa_groups(ctx):
        if only_groups is not None and g["id"] not in only_groups:
            continue
        todo = [o for o in g["outputs"] if (noaa.output_id(o), o["ver"]) not in done]
        if not todo:
            log("skip", g["id"], day.isoformat(), "complete")
            continue
        sup = None
        if supersedes_by_output:
            sup = next((supersedes_by_output.get(noaa.output_id(o)) for o in todo
                        if supersedes_by_output.get(noaa.output_id(o))), None)
        jobs.append((g, todo, sup))
    results = {}
    with cf.ThreadPoolExecutor(max_workers=1) as side:
        later = {g["id"]: side.submit(_run_group, ctx, g, day, todo, sup) for g, todo, sup in jobs
                 if g.get("overlap")}
        for g, todo, sup in jobs:
            if g["id"] not in later:
                results[g["id"]] = _run_group(ctx, g, day, todo, sup)
        for gid, f in later.items():
            results[gid] = f.result()
    for g, _, _ in jobs:
        res = results[g["id"]]
        book.add_group_result(day, res)
        for r in res.reasons:
            log("day-problem", g["id"], day.isoformat(), r)


def lane_noaa(ctx, start, end, pilot):
    for day in _days(start, end):
        if not ctx.within_budget():
            log("budget", "stopping before", day.isoformat())
            break
        ctx.breaker.check_before_day()
        book = Book(ctx)
        run_noaa_day(ctx, book, day)
        finish_day(ctx, book, day, pilot)


def _iem_enabled(ctx):
    cut_ids = {c.get("iem_dataset") for c in ctx.breaker.active_cuts() if c.get("iem_dataset")}
    return [d for d in ctx.cfg["iem"]["datasets"] if d.get("enabled") and d["id"] not in cut_ids]


def lane_iem(ctx, start, end, kind):
    """kind: iem (observations + other sources), vtec (warnings only), pilot (all, day shape)."""
    book = Book(ctx)
    today = utcnow().date()
    chunked = kind != "pilot"
    catalogs = {}
    days = list(_days(start, end))
    mans = {d: complete_set(day_manifests(ctx.store, d)) for d in days}
    for ds in _iem_enabled(ctx):
        is_vtec = bool(ds.get("vtec"))
        if kind == "vtec" and not is_vtec:
            continue
        if kind == "iem" and is_vtec:
            continue
        ctx.breaker.check_before_day()
        skip = frozenset(d for d in days if (ds["id"], ds["ver"]) in mans[d])
        if ds["kind"] == "catalog":
            catalogs = iem.run_catalog(ctx, book, ds, today)
        elif ds["kind"] == "table":
            if (ds.get("station_list") or ds.get("skip_empty_catalog")) and not catalogs:
                cat = next(c for c in ctx.cfg["iem"]["datasets"] if c["kind"] == "catalog")
                catalogs = iem.run_catalog(ctx, book, cat, today)
            iem.run_table(ctx, book, ds, start, end, ds.get("chunk", "day") if chunked else "day", catalogs, skip)
        elif ds["kind"] == "asserved":
            iem.run_asserved(ctx, book, ds, start, end, skip)
    if kind in ("iem", "pilot"):
        for ds in ctx.cfg["other"]["datasets"]:
            if not ds.get("enabled"):
                continue
            ctx.breaker.check_before_day()
            if ds["kind"] == "listing":
                other.run_listing(ctx, book, ds, today)
            elif ds["kind"] == "daily-files":
                skip = frozenset(d for d in days if (ds["id"], ds["ver"]) in mans[d])
                other.run_daily_files(ctx, book, ds, start, end, skip)
    for day in book.days():
        finish_day(ctx, book, day, kind == "pilot")


def lane_catalog(ctx):
    """Station catalogues only; runs before the gridded and observation lanes, which read it."""
    book = Book(ctx)
    iem.run_catalog(ctx, book, iem.catalog_dataset(ctx.cfg), utcnow().date())
    failed = False
    for day in book.days():
        failed = finish_day(ctx, book, day, False) or failed
    return failed


def verify_datasets(cfg):
    out = []
    for g in cfg["noaa"]["groups"]:
        for o in noaa.active_outputs(g):
            if o["kind"] == "tile":
                out.append((o["layer"], o["ver"], "tile"))
                out.extend((ov["layer"], o["ver"], "tile") for ov in o.get("overviews", []))
            else:
                out.append((o["dataset"], o["ver"], "src"))
    out += [(d["id"], d["ver"], "src") for d in cfg["iem"]["datasets"]]
    out += [(d["id"], d["ver"], "src") for d in cfg["other"]["datasets"]]
    out.append(("boxflags-vtec", "v1", "src"))
    return out


def lane_verify(ctx, days, check_all=False):
    vcfg = ctx.cfg.get("verify", {})
    total_fail = 0
    for day in days:
        rep = verify.verify_day(ctx.store, day, verify_datasets(ctx.cfg), vcfg.get("sample_share", 0.02),
                                vcfg.get("min_per_day", 3), seed=f"{ctx.run['id']}-{day.isoformat()}",
                                check_all=check_all)
        rep["run"] = {k: ctx.run[k] for k in ("id", "attempt", "job", "git_sha")}
        if ctx.store.write:
            ctx.store.put(vrf_key(day, ctx.run["id"], ctx.run["attempt"], ctx.job), canonical_json(rep),
                          "application/json")
        else:
            print(canonical_json(rep).decode("ascii"), flush=True)
        total_fail += verify.failures(rep)
        ctx.breaker.after_day(day, 0, verify.failures(rep) > 0, verify.checked(rep), verify.failures(rep))
    return total_fail


def _days(a, b):
    d = a
    while d <= b:
        yield d
        d += _dt.timedelta(days=1)


# ---------------------------------------------------------------- main

def main(env=None):
    env = env if env is not None else os.environ
    lane = env.get("LANE", "")
    kind = env.get("RUN_KIND", lane)
    if lane == "catalog" and not env.get("CHUNK"):
        today = utcnow().date()
        start = end = today
    else:
        try:
            start, end = parse_chunk(env.get("CHUNK", ""))
        except (ValueError, AttributeError):
            print("run: CHUNK must be YYYY-MM-DD..YYYY-MM-DD", file=sys.stderr)
            return 2
    if lane not in ("noaa", "iem", "vtec", "verify", "catalog"):
        print("run: unknown LANE", file=sys.stderr)
        return 2
    cfg = load_config()
    write = _env_bool("WRITE", False, env) if lane != "verify" else True
    store = Store.from_env(write=write if lane != "verify" else False)
    if lane == "verify":
        store.write = store.online
    job = f"{lane}-{start.isoformat()}"
    ctx = Ctx(cfg, "pilot" if kind == "pilot" else lane, job, store)
    log("start", lane, f"kind={kind}", f"chunk={start}..{end}", f"write={store.write}", f"online={store.online}")
    code = 0
    try:
        if lane == "noaa":
            lane_noaa(ctx, start, end, pilot=(kind == "pilot"))
        elif lane in ("iem", "vtec"):
            lane_iem(ctx, start, end, "pilot" if kind == "pilot" else lane)
        elif lane == "catalog":
            code = 1 if lane_catalog(ctx) else 0
        else:
            code = 1 if lane_verify(ctx, list(_days(start, end))) else 0
    except BreakerTripped as tr:
        code = EXIT_TRIPPED if tr.origin else 0
    except FATAL as err:
        log("FATAL", type(err).__name__, str(err)[:200])
        try:
            ctx.breaker.trip(f"fatal: {type(err).__name__}")
        except BreakerTripped:
            pass
        code = EXIT_TRIPPED
    except Exception as err:  # anything else: type only, so no endpoint, bucket or row text reaches the public log
        log("FATAL", type(err).__name__)
        try:
            ctx.breaker.trip(f"fatal: {type(err).__name__}")
        except BreakerTripped:
            pass
        code = EXIT_TRIPPED
    finally:
        ctx.close()
    log("end", lane, f"exit={code}", f"puts={store.puts}", f"bytes={store.bytes_written}")
    return code


if __name__ == "__main__":
    sys.exit(main())
