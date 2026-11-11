"""Other public sources, fetched sequentially: directory listings (new files only, or new content), and fixed
daily file names. Everything is kept as served (already-compressed files as is, text zstd-compressed)."""

import datetime as _dt
import re

from . import compact, log
from .manifest import iso
from .store import src_key


def _names_in_store(ctx, ds):
    if not ctx.store.online:
        return set()
    return {o["key"].rsplit("/", 1)[1] for o in ctx.store.list(f"src/{ds['id']}/{ds['ver']}/")}


def _stored_name(name, sha=None):
    base = name if sha is None else f"{name}.{sha[:16]}"
    return base if name.endswith(".gz") else base + ".zst"


def _pack(name, body):
    if name.endswith(".gz"):
        return body, "application/gzip"
    return compact.zstd(body), "application/zstd"


def latest_index(parent_body, ds):
    """The index URL under a versioned parent: the highest version directory matching index_from.dir (its groups are
    the version numbers), then index_from.sub. None when no directory matches."""
    spec = ds["index_from"]
    rx = re.compile(spec["dir"])
    found = {m.group(0): tuple(int(g) for g in m.groups()) for m in rx.finditer(parent_body.decode("utf-8", "replace"))}
    if not found:
        return None
    best = max(found, key=lambda k: found[k])
    return spec["parent"] + best + spec["sub"]


def run_listing(ctx, book, ds, today):
    entry = book.entry(today, ds["id"], ds["ver"])
    index = ds.get("index")
    if ds.get("index_from"):
        pr = ctx.http.fetch(ds["index_from"]["parent"])
        entry["steps"]["expected"] += 1
        if not pr.ok:
            entry.add_missing(iso(ctx.started), pr.status, pr.tries)
            return
        entry["steps"]["got"] += 1
        index = latest_index(pr.body, ds)
        if index is None:
            entry.add_missing(iso(ctx.started), 404, 1)
            return
    r = ctx.http.fetch(index)
    entry["steps"]["expected"] += 1
    if not r.ok:
        entry.add_missing(iso(ctx.started), r.status, r.tries)
        return
    entry["steps"]["got"] += 1
    rx = re.compile(ds["pattern"])
    names = sorted({m.group(0) for m in rx.finditer(r.body.decode("utf-8", "replace"))})
    if ds.get("min_year"):
        names = [n for n in names if not rx.fullmatch(n).groups() or int(rx.fullmatch(n).group(1)) >= ds["min_year"]]
    have = _names_in_store(ctx, ds)
    new = 0
    for name in names:
        if not ds.get("by_content") and _stored_name(name) in have:
            continue
        if not ctx.within_budget():
            break
        fr = ctx.http.fetch(index + name)
        entry["steps"]["expected"] += 1
        if not fr.ok:
            entry.add_missing(iso(ctx.started), fr.status, fr.tries)
            continue
        entry["steps"]["got"] += 1
        stored = _stored_name(name, fr.sha256 if ds.get("by_content") else None)
        if stored in have:
            continue
        entry.add_source(**fr.source_record())
        data, ctype = _pack(name, fr.body)
        if ctype == "application/zstd":
            entry["checks"]["roundtrip"] = "pass" if compact.unzstd(data) == fr.body else "fail"
        book.put(today, entry, src_key(ds["id"], ds["ver"], today, ctx.vintage, stored), data, ctype)
        new += 1
    if new == 0 and not entry["steps"]["missing"]:
        entry["status"] = "unchanged"
        book.unchanged += 1
    log("listing", ds["id"], f"files={len(names)}", f"new={new}")


def run_daily_files(ctx, book, ds, start, end, skip=frozenset()):
    """Fixed per-day file names (one URL per suffix)."""
    d = start
    while d <= end:
        if d not in skip and ctx.within_budget():
            entry = book.entry(d, ds["id"], ds["ver"])
            when = _dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc)
            for suffix in ds["suffixes"]:
                url = ds["url"].format(yymmdd=when.strftime("%y%m%d"), suffix=suffix)
                r = ctx.http.fetch(url)
                entry["steps"]["expected"] += 1
                entry.add_source(**r.source_record(valid_utc=iso(when)))
                if not r.ok:
                    entry.add_missing(iso(when), r.status, r.tries)
                    continue
                entry["steps"]["got"] += 1
                name = url.rsplit("/", 1)[1]
                data, ctype = _pack(name, r.body)
                entry["checks"]["roundtrip"] = "pass" if compact.unzstd(data) == r.body else "fail"
                book.put(d, entry, src_key(ds["id"], ds["ver"], d, ctx.vintage, _stored_name(name)), data, ctype)
        d += _dt.timedelta(days=1)
