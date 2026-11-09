"""Iowa Environmental Mesonet (IEM) sources, fetched sequentially with per-endpoint spacing.

Kinds (config ``iem.datasets``):
- ``catalog``: station network GeoJSON; a new object only when its sha256 changes;
- ``table``: delimited replies per network and time window -> Parquet (rt) day slices (fallback: as served, zstd);
- ``asserved``: one reply per UTC day window, kept as served (zip as is, text zstd-compressed).
Backfill uses the configured window (``chunk``: month or day); daily runs and refreshes use day windows.
"""

import datetime as _dt
import hashlib
import json
import urllib.parse

from . import compact, log, stormbox
from .manifest import iso
from .store import src_key

DAY = _dt.timedelta(days=1)


# ---------------------------------------------------------------- helpers

def networks(cfg_iem, ds):
    out = []
    for pat in ds.get("networks", []):
        if "{ST}" in pat:
            out.extend(pat.replace("{ST}", st) for st in cfg_iem["states"])
        else:
            out.append(pat)
    return out


def windows(start, end, chunk):
    """[(t0, t1)) datetime windows covering days start..end."""
    out = []
    d = start
    while d <= end:
        if chunk == "month":
            nxt = (d.replace(day=1) + _dt.timedelta(days=32)).replace(day=1)
            last = min(end, nxt - DAY)
        else:
            last = d
        t0 = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
        t1 = _dt.datetime(last.year, last.month, last.day, tzinfo=_dt.timezone.utc) + DAY
        out.append((t0, t1))
        d = last + DAY
    return out


def time_params(style, t0, t1):
    if style == "ymdhm":
        return [("year1", t0.year), ("month1", t0.month), ("day1", t0.day), ("hour1", t0.hour),
                ("minute1", t0.minute), ("year2", t1.year), ("month2", t1.month), ("day2", t1.day),
                ("hour2", t1.hour), ("minute2", t1.minute)]
    if style == "ymd":
        e = t1 - DAY  # inclusive end day
        return [("year1", t0.year), ("month1", t0.month), ("day1", t0.day), ("year2", e.year), ("month2", e.month),
                ("day2", e.day)]
    if style == "sts_ets":
        return [("sts", t0.strftime("%Y-%m-%dT%H:%MZ")), ("ets", t1.strftime("%Y-%m-%dT%H:%MZ"))]
    raise ValueError(f"unknown time style {style}")


def build_url(base, path, params):
    return base + path + "?" + urllib.parse.urlencode([(k, str(v)) for k, v in params])


def parse_catalog(data):
    """[(station id, lat, lon)] from a network GeoJSON."""
    doc = json.loads(data.decode("utf-8"))
    out = []
    for f in doc.get("features", []):
        sid = f.get("id") or (f.get("properties") or {}).get("sid")
        geom = f.get("geometry") or {}
        if sid and geom.get("type") == "Point":
            lon, lat = geom["coordinates"][:2]
            out.append((str(sid), float(lat), float(lon)))
    return out


# ---------------------------------------------------------------- catalog

def _existing_names(ctx, ds):
    if not ctx.store.online:
        return set()
    return {o["key"].rsplit("/", 1)[1] for o in ctx.store.list(f"src/{ds['id']}/{ds['ver']}/")}


def run_catalog(ctx, book, ds, today):
    """Pulls every network catalogue; returns {network: [(id, lat, lon)]}."""
    out = {}
    have = _existing_names(ctx, ds)
    nets = networks(ctx.cfg["iem"], ds)
    entry = book.entry(today, ds["id"], ds["ver"])
    entry["steps"]["expected"] += len(nets)
    changed = 0
    for net in nets:
        url = build_url(ctx.cfg["iem"]["base"], ds["path"].format(net=net), [])
        url = url.rstrip("?")
        r = ctx.http.fetch(url)
        ctx.breaker.observe_http(ctx.http)
        if not r.ok:
            entry.add_missing(iso(ctx.started), r.status, r.tries)
            continue
        entry["steps"]["got"] += 1
        try:
            out[net] = parse_catalog(r.body)
        except (ValueError, KeyError):
            entry.add_missing(iso(ctx.started), r.status, r.tries)
            continue
        name = f"{net}-{r.sha256[:16]}.geojson.zst"
        if name in have:
            continue
        entry.add_source(**r.source_record())
        data = compact.zstd(r.body)
        ok = compact.unzstd(data) == r.body
        entry["checks"]["roundtrip"] = "pass" if ok and entry["checks"].get("roundtrip") != "fail" else "fail"
        book.put(today, entry, src_key(ds["id"], ds["ver"], today, ctx.vintage, name), data, "application/zstd")
        changed += 1
    if changed == 0 and entry["steps"]["got"] == entry["steps"]["expected"]:
        entry["status"] = "unchanged"
        book.unchanged += 1
    log("catalog", ds["id"], f"networks={len(nets)}", f"new={changed}")
    return out


def station_points_from_store(store, network_suffix="_ASOS"):
    """[(lat, lon)] of every station in the newest stored catalogue of each *_ASOS network, or None."""
    if not store.online:
        return None
    newest = {}
    for o in store.list("src/iem-stations/v1/"):
        name = o["key"].rsplit("/", 1)[1]
        net = name.split("-")[0]
        if not net.endswith(network_suffix):
            continue
        vint = o["key"].split("/")[-2]
        if net not in newest or vint > newest[net][0]:
            newest[net] = (vint, o["key"])
    if not newest:
        return None
    pts = []
    for _, key in sorted(newest.values(), key=lambda x: x[1]):
        data, _ = store.get(key)
        pts.extend((lat, lon) for _, lat, lon in parse_catalog(compact.unzstd(data)))
    return pts


# ---------------------------------------------------------------- tables

def _day_sha(table):
    return hashlib.sha256(compact.table_to_bytes(table)).hexdigest()


def _store_table_reply(ctx, book, ds, net, r, t0, t1, days_wanted):
    """Stores one reply as Parquet (rt) day slices. Returns False if it is not round-trippable."""
    try:
        table = compact.csv_to_table(r.body)
        slices = compact.slice_by_day(table, ds["time_column"])
    except compact.NotRoundTrippable as err:
        log("not-rt", ds["id"], net, type(err).__name__)  # never the message: it can quote a data row
        return False
    parts = []
    for dstr in sorted(slices):
        day = _dt.date.fromisoformat(dstr)
        if day not in days_wanted:
            # a row outside the requested window (e.g. local-day files): kept with the first requested day
            continue
        parts.append((day, slices[dstr]))
    outside = [d for d in slices if _dt.date.fromisoformat(d) not in days_wanted]
    if outside:
        log("rows-outside-window", ds["id"], net, f"days={len(outside)}")
        return False
    blobs = [(day, compact.table_to_parquet(t), t) for day, t in parts]
    if blobs:
        ok = compact.roundtrip_ok([b for _, b, _ in blobs], r.sha256)
    else:
        ok = compact.table_to_bytes(table) == r.body
    for day, blob, t in blobs:
        entry = book.entry(day, ds["id"], ds["ver"])
        entry["checks"]["roundtrip"] = "pass" if ok and entry["checks"].get("roundtrip") != "fail" else "fail"
        entry["checks"]["rows"] = int(entry["checks"].get("rows", 0)) + t.num_rows
        book.put(day, entry, src_key(ds["id"], ds["ver"], day, ctx.vintage, f"{net}.parquet"), blob,
                 "application/vnd.apache.parquet", covers={"network": net, "day_sha256": _day_sha(t)})
    if not ok:
        book.failed_days.update(days_wanted)
    return True


def _store_asserved_table(ctx, book, ds, net, r, day):
    entry = book.entry(day, ds["id"], ds["ver"])
    data = compact.zstd(r.body)
    ok = compact.unzstd(data) == r.body
    entry["checks"]["roundtrip"] = "pass" if ok else "fail"
    book.put(day, entry, src_key(ds["id"], ds["ver"], day, ctx.vintage, f"{net}.csv.zst"), data, "application/zstd",
             covers={"network": net, "as_served": True})


def run_table(ctx, book, ds, start, end, chunk, catalogs, skip=frozenset()):
    """One dataset over days start..end. ``skip``: days already complete."""
    nets = networks(ctx.cfg["iem"], ds)
    days_all = [d for d in _days(start, end) if d not in skip]
    if not days_all:
        return
    for d in days_all:
        book.entry(d, ds["id"], ds["ver"])["steps"]["expected"] += len(nets)
    for net in nets:
        stations = None
        if ds.get("station_list"):
            stations = [s for s, _, _ in catalogs.get(net, [])]
            if not stations:
                for d in days_all:
                    book.entry(d, ds["id"], ds["ver"]).add_missing(f"{d.isoformat()}T00:00:00Z", 0, 0)
                continue
        for t0, t1 in windows(days_all[0], days_all[-1], chunk):
            wdays = {d for d in _days(t0.date(), (t1 - DAY).date()) if d in set(days_all)}
            if not wdays:
                continue
            if not ctx.within_budget():
                return
            got = _request_table(ctx, book, ds, net, stations, t0, t1, wdays)
            if got == "not-rt" and len(wdays) > 1:
                for d in sorted(wdays):
                    td0 = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
                    if _request_table(ctx, book, ds, net, stations, td0, td0 + DAY, {d}) == "not-rt":
                        pass


def _days(a, b):
    d = a
    while d <= b:
        yield d
        d += DAY


def _request_table(ctx, book, ds, net, stations, t0, t1, wdays):
    params = []
    if stations is not None:
        params += [("station", s) for s in stations]
    if not ds.get("station_list"):
        params.append(("network", net))
    params += [tuple(p) for p in ds.get("params", [])]
    params += time_params(ds["time_params"], t0, t1)
    url = build_url(ctx.cfg["iem"]["base"], ds["path"], params)
    r = ctx.http.fetch(url)
    ctx.breaker.observe_http(ctx.http)
    src = r.source_record(valid_utc=iso(t0))
    for d in wdays:
        e = book.entry(d, ds["id"], ds["ver"])
        e.add_source(**src)
        if not r.ok:
            e.add_missing(f"{d.isoformat()}T00:00:00Z", r.status, r.tries)
    if not r.ok:
        return "missing"
    if _store_table_reply(ctx, book, ds, net, r, t0, t1, wdays):
        for d in wdays:
            book.entry(d, ds["id"], ds["ver"])["steps"]["got"] += 1
        return "ok"
    if len(wdays) == 1:
        _store_asserved_table(ctx, book, ds, net, r, next(iter(wdays)))
        book.entry(next(iter(wdays)), ds["id"], ds["ver"])["steps"]["got"] += 1
        return "as-served"
    return "not-rt"


# ---------------------------------------------------------------- as served (LSR, warnings, text)

def run_asserved(ctx, book, ds, start, end, skip=frozenset()):
    h0, h1 = ds.get("window_h", [0, 24])
    for day in _days(start, end):
        if day in skip:
            continue
        if not ctx.within_budget():
            return
        t = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc)
        t0, t1 = t + _dt.timedelta(hours=h0), t + _dt.timedelta(hours=h1)
        url = build_url(ctx.cfg["iem"]["base"], ds["path"], [tuple(p) for p in ds.get("params", [])] +
                        time_params(ds["time_params"], t0, t1))
        r = ctx.http.fetch(url)
        ctx.breaker.observe_http(ctx.http)
        entry = book.entry(day, ds["id"], ds["ver"])
        entry["steps"]["expected"] += 1
        entry.add_source(**r.source_record(valid_utc=iso(t0)))
        if not r.ok:
            entry.add_missing(iso(t0), r.status, r.tries)
            continue
        entry["steps"]["got"] += 1
        store_asserved(ctx, book, ds, day, entry, r.body)


def store_asserved(ctx, book, ds, day, entry, body):
    ymd = day.strftime("%Y%m%d")
    if ds["ext"] == "zip":
        data, name, ctype = body, f"{ds['id']}-{ymd}.zip", "application/zip"
        entry["checks"]["roundtrip"] = "pass"
    else:
        data = compact.zstd(body)
        name, ctype = f"{ds['id']}-{ymd}.{ds['ext']}.zst", "application/zstd"
        entry["checks"]["roundtrip"] = "pass" if compact.unzstd(data) == body else "fail"
    book.put(day, entry, src_key(ds["id"], ds["ver"], day, ctx.vintage, name), data, ctype)
    if ds.get("vtec") and "svto" in ds["id"]:
        try:
            boxes = stormbox.polygon_rule(stormbox.polygons_from_zip(body))
        except Exception as err:  # malformed archive: keep the raw file, flag nothing
            log("vtec-parse", day.isoformat(), type(err).__name__)
            return
        flags = book.entry(day, "boxflags-vtec", "v1")
        flags["steps"]["expected"] += 1
        flags["steps"]["got"] += 1
        flags["sources"].append({"url": entry["sources"][-1]["url"], "range": None,
                                 "sha256": entry["sources"][-1]["sha256"], "etag": None, "last_modified": None,
                                 "valid_utc": None, "http": 200, "tries": 1})
        doc = {"schema": "box-flags/1", "day": day.isoformat(), "rule": "warning_polygons", "min_polygons": 2,
               "boxes": boxes}
        from . import canonical_json

        book.put(day, flags, src_key("boxflags-vtec", "v1", day, ctx.vintage, f"boxflags-vtec-{ymd}.json"),
                 canonical_json(doc), "application/json")


# ---------------------------------------------------------------- refresh (re-pull a past day)

def refresh_day(ctx, book, ds, day, newest, catalogs):
    """Re-pulls ``day`` in day shape. Writes a full new vintage only if anything changed."""
    if ds["kind"] == "asserved":
        h0, h1 = ds.get("window_h", [0, 24])
        t = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc)
        t0, t1 = t + _dt.timedelta(hours=h0), t + _dt.timedelta(hours=h1)
        url = build_url(ctx.cfg["iem"]["base"], ds["path"], [tuple(p) for p in ds.get("params", [])] +
                        time_params(ds["time_params"], t0, t1))
        r = ctx.http.fetch(url)
        ctx.breaker.observe_http(ctx.http)
        if not r.ok:
            return "missing"
        old = {s.get("sha256") for s in (newest or {}).get("sources", [])}
        if r.sha256 in old:
            book.unchanged += 1
            return "unchanged"
        entry = book.entry(day, ds["id"], ds["ver"])
        entry["supersedes"] = newest["vintage"] if newest else None
        entry["steps"]["expected"] += 1
        entry["steps"]["got"] += 1
        entry.add_source(**r.source_record(valid_utc=iso(t0)))
        store_asserved(ctx, book, ds, day, entry, r.body)
        return "changed"
    # tables: compare each network's day slice with the newest vintage
    old = {o.get("covers", {}).get("network"): o.get("covers", {}).get("day_sha256")
           for o in (newest or {}).get("objects", [])}
    staged = book.child()
    run_table(ctx, staged, ds, day, day, "day", catalogs)
    new = staged.pending_covers()
    if newest is not None and new == old:
        book.unchanged += 1
        return "unchanged"
    staged.commit_into(book, supersedes=newest["vintage"] if newest else None)
    return "changed"
