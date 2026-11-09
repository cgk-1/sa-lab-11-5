"""NOAA Open Data Dissemination (AWS) sources: .idx lookup, GRIB2 header walk when there is no .idx, byte-range
reads, S3 listing, ETag / Last-Modified capture; and the per-day processing of one source group into sparse cubes,
display tiles, as-served bundles and box flags."""

import concurrent.futures as cf
import copy
import datetime as _dt
import gzip
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

from . import canonical_json, compact, grib, log, sparse, stormbox, tile
from .net import HttpError
from .manifest import DatasetEntry, iso
from .store import src_key, tile_key

NODD_WORKERS = 8


# ---------------------------------------------------------------- addressing

def fill(template, when):
    return template.format(ymd=when.strftime("%Y%m%d"), HH=when.strftime("%H"), MM=when.strftime("%M"),
                           Y=when.strftime("%Y"), m=when.strftime("%m"), d=when.strftime("%d"),
                           yymmdd=when.strftime("%y%m%d"))


def nodd_url(bucket, key):
    return f"https://{bucket}.s3.amazonaws.com/{urllib.parse.quote(key)}"


def day_steps(day, cadence_min):
    t0 = _dt.datetime(day.year, day.month, day.day, tzinfo=_dt.timezone.utc)
    return [t0 + _dt.timedelta(minutes=m) for m in range(0, 1440, int(cadence_min))]


# ---------------------------------------------------------------- .idx

class IdxEntry:
    def __init__(self, n, offset, end, name):
        self.n, self.offset, self.end, self.name = n, offset, end, name


def parse_idx(text):
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(":")
        if len(parts) < 6:
            continue
        rows.append((int(parts[0].split(".")[0]), int(parts[1]), ":".join(parts[3:6])))
    out = []
    for k, (n, off, name) in enumerate(rows):
        end = rows[k + 1][1] - 1 if k + 1 < len(rows) else None
        out.append(IdxEntry(n, off, end, name))
    return out


def find_entry(entries, pattern):
    hits = [e for e in entries if e.name == pattern]
    if len(hits) > 1:
        raise grib.GribError(f"ambiguous .idx pattern: {pattern}")
    return hits[0] if hits else None


# ---------------------------------------------------------------- listing

def list_keys(http, bucket, prefix):
    out, token = [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        r = http.get(f"https://{bucket}.s3.amazonaws.com/?" + urllib.parse.urlencode(q))
        if r.status == 404:
            return out
        root = ET.fromstring(r.body)
        for c in root.findall(f"{ns}Contents"):
            out.append({"key": c.findtext(f"{ns}Key"), "etag": (c.findtext(f"{ns}ETag") or "").strip('"'),
                        "bytes": int(c.findtext(f"{ns}Size") or 0)})
        if (root.findtext(f"{ns}IsTruncated") or "false").lower() != "true":
            return out
        token = root.findtext(f"{ns}NextContinuationToken")


# ---------------------------------------------------------------- header walk

def walk_headers(http, url, chunk=4096, max_messages=1000):
    """[header dict with offset/length] by ranged reads of each message start, or (None, status) on 404."""
    out, off, total = [], 0, None
    while len(out) < max_messages:
        if total is not None and off >= total:
            break
        r = http.get(url, rng=(off, off + chunk - 1), allow=(404, 416))
        if r.status == 404:
            return None, 404
        if r.status == 416 or not r.body:
            break
        cr = r.headers.get("content-range", "")
        if "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            total = int(cr.rsplit("/", 1)[1])
        buf = r.body
        try:
            h = grib.parse_header(buf)
        except grib.NeedMore as nm:
            buf = http.get(url, rng=(off, off + nm.n + chunk)).body
            h = grib.parse_header(buf)
        h["offset"] = off
        out.append(h)
        off += h["length"]
        if total is None and len(r.body) < chunk:
            break
    return out, 200


# ---------------------------------------------------------------- fetching one step

def _fetch_from(http, url, idx_suffix, pattern, field, cache):
    """Response for one field from one GRIB file, or (None, status, tries)."""
    ckey = ("idx", url)
    if ckey not in cache:
        cache[ckey] = http.fetch(url + idx_suffix) if idx_suffix else None
    ir = cache[ckey]
    if ir is not None and ir.ok:
        e = find_entry(parse_idx(ir.body.decode("ascii", "replace")), pattern)
        if e is None:
            return None, 404, ir.tries
        r = http.fetch(url, rng=(e.offset, e.end))
        return (r, r.status, r.tries) if r.ok else (None, r.status, r.tries)
    wkey = ("walk", url)
    if wkey not in cache:
        try:
            cache[wkey] = walk_headers(http, url)
        except HttpError as err:
            cache[wkey] = (None, err.status or 0)
    headers, status = cache[wkey]
    if headers is None:
        return None, status, 1
    hits = [h for h in headers if grib.matches(h, field)]
    if len(hits) != 1:
        return None, 404 if not hits else 409, 1
    h = hits[0]
    r = http.fetch(url, rng=(h["offset"], h["offset"] + h["length"] - 1))
    return (r, r.status, r.tries) if r.ok else (None, r.status, r.tries)


def fetch_grib_step(http, group, fields, when):
    """{field: (Response|None, http, tries, fallback flag)} for one step."""
    url = nodd_url(group["bucket"], fill(group["key"], when))
    cache, out = {}, {}
    for name in fields:
        f = group["fields"][name]
        r, st, tr = _fetch_from(http, url, group.get("idx_suffix"), f["idx"], f, cache)
        fb = None
        if r is None and f.get("fallback"):
            fbc = f["fallback"]
            furl = nodd_url(group["bucket"], fill(fbc["key"], when))
            r, st2, tr2 = _fetch_from(http, furl, group.get("idx_suffix"), fbc["idx"], f, cache)
            if r is not None:
                fb, st, tr = fbc["flag"], st2, tr2
        out[name] = (r, st, tr, fb)
    return out


# ---------------------------------------------------------------- cuts

def apply_cuts(group, cuts):
    g = copy.deepcopy(group)
    for c in cuts:
        if c.get("group") != g["id"]:
            continue
        layers = c.get("layers")
        for o in g["outputs"]:
            if layers is None or o.get("layer") in layers:
                o["ver"] = f"{o['ver'].split('-')[0]}-{c['ver_suffix']}"
                if "every_n_steps" in c.get("set", {}):
                    o["every_n_steps"] = c["set"]["every_n_steps"]
        if layers is None:
            for k, v in c.get("set", {}).items():
                if k != "every_n_steps":
                    g[k] = v
    return g


def output_id(o):
    return o.get("dataset") or o.get("layer")


# ---------------------------------------------------------------- processing

class GroupResult:
    def __init__(self):
        self.entries = []
        self.written = []  # (put_result, bytes) for in-job verification
        self.failed = False
        self.reasons = []
        self.bytes = 0


class _Out:
    """One output of a group during a day."""

    def __init__(self, spec, entry, steps_used):
        self.spec = spec
        self.entry = entry
        self.steps_used = steps_used
        self.obj = None
        self.daymax = None


def _era(group, day):
    return [e["flag"] for e in group.get("era", []) if day.isoformat() < e["before"]]


def _put(ctx, res, entry, key, data, ctype, covers=None):
    pr = ctx.store.put(key, data, ctype)
    entry.add_object(pr, covers)
    res.written.append((pr, data))
    res.bytes += pr["bytes"]
    return pr


def _tile_revision_key(ctx, layer, ver, day, supersedes):
    if supersedes is None:
        return tile_key(layer, ver, day, 0)
    for k in range(1, 100):
        key = tile_key(layer, ver, day, k)
        if ctx.store.head(key) is None:
            return key
    raise RuntimeError("no free tile revision")


def _finish_tile(ctx, res, out, cube, day, grid, supersedes, layer=None, ver=None):
    layer = layer or out.spec["layer"]
    ver = ver or out.spec["ver"]
    path = os.path.join(ctx.staging, f"{layer}-{day.isoformat()}.pmtiles")
    bounds = None
    if grid is not None:
        try:
            lats, lons = grid.latlons()
            bounds = (float(np.nanmin(lons)), float(np.nanmin(lats)), float(np.nanmax(lons)), float(np.nanmax(lats)))
        except grib.GribError:
            bounds = None
    meta = {"layer": layer, "ver": ver, "day": day.isoformat(), "source": out.spec.get("field") or
            out.spec.get("product") or "", "grid": list(grid.signature) if grid is not None and grid.signature else None}
    cube.write(path, meta, bounds)
    ok = cube.decode_back_ok and cube.verify_file(path)
    with open(path, "rb") as fh:
        data = fh.read()
    os.unlink(path)
    key = _tile_revision_key(ctx, layer, ver, day, supersedes)
    out.entry["checks"]["decode_back"] = "pass" if ok else "fail"
    _put(ctx, res, out.entry, key, data, "application/vnd.pmtiles")
    side = cube.sidecar_parquet()
    if side is not None:
        _put(ctx, res, out.entry, key[: -len(".pmtiles")] + ".overflow.parquet", side, "application/vnd.apache.parquet")
    if not ok:
        res.failed = True
        res.reasons.append(f"decode-back failed: {layer}")


def process_group(ctx, group, day, outputs, supersedes=None):
    """Fetches and builds ``outputs`` (subset of the group's outputs) for ``day``."""
    res = GroupResult()
    kind = group["kind"]
    if kind == "grib":
        _process_grib(ctx, group, day, outputs, supersedes, res)
    elif kind == "files":
        _process_files(ctx, group, day, outputs, supersedes, res)
    elif kind == "listed":
        _process_listed(ctx, group, day, outputs, supersedes, res)
    elif kind == "listed-all":
        _process_listed_all(ctx, group, day, outputs, supersedes, res)
    else:
        raise ValueError(f"unknown group kind {kind}")
    for e in res.entries:
        e.finish_status()
        if e["checks"].get("decode_back") == "fail" or e["checks"].get("roundtrip") == "fail":
            e["status"] = "partial"
    return res


def _new_entry(ctx, o, group, day, supersedes):
    e = DatasetEntry(output_id(o), o["ver"], ctx.vintage)
    e["era"] = _era(group, day)
    e["supersedes"] = supersedes
    return e


def _station_cells(ctx, grid):
    pts = ctx.station_points()
    if pts is None:
        return None
    return [grid.locate(lat, lon) for lat, lon in pts]


def _init_outputs(ctx, group, day, outputs, n_steps, grid, supersedes, res):
    outs = []
    for o in outputs:
        e = _new_entry(ctx, o, group, day, supersedes)
        res.entries.append(e)
        every = int(o.get("every_n_steps", 1))
        used = [i for i in range(n_steps) if i % every == 0]
        e["steps"]["expected"] = len(used)
        outs.append(_Out(o, e, set(used)))
    return outs


def _feed(ctx, outs, field_key, i, values, grid, n_steps, cubes):
    for out in outs:
        o = out.spec
        if (o.get("field") or o.get("product")) != field_key:
            continue
        if i not in out.steps_used:
            continue
        k = id(out)
        if o["kind"] == "sparse":
            if k not in cubes:
                cells = _station_cells(ctx, grid) if o.get("station_points") else None
                cubes[k] = sparse.SparseCube(values.shape, n_steps, o["threshold_mph"] / sparse.MPH_PER_MS,
                                             o.get("lattice_step", 10), cells)
                out.entry["checks"]["station_points"] = len([c for c in cells if c is not None]) \
                    if cells is not None else "unavailable"
            cubes[k].add_step(i, values)
        elif o["kind"] == "tile":
            if k not in cubes:
                enc = tile.Encoding(ctx.cfg["encodings"][o["enc"]])
                cubes[k] = tile.TileCube(values.shape, n_steps, enc, ctx.staging, ctx.cfg.get("tile_size", 128))
            cubes[k].add_step(i, values)
        elif o["kind"] == "boxflags":
            out.daymax = values.copy() if out.daymax is None else np.fmax(out.daymax, values)
        out.entry["steps"]["got"] += 1


def _finalize(ctx, group, day, outs, cubes, grid, supersedes, res, step_labels):
    ymd = day.strftime("%Y%m%d")
    for out in outs:
        o = out.spec
        k = id(out)
        if o["kind"] == "sparse" and k in cubes:
            cube = cubes[k]
            data, rows = cube.to_parquet({"dataset": o["dataset"], "day": day.isoformat(), "field": o["field"],
                                          "step_labels": step_labels, "unit": "m/s"})
            ok = sparse.decode_check(data, cube)
            out.entry["checks"].update({"decode_back": "pass" if ok else "fail", "rows": rows})
            _put(ctx, res, out.entry, src_key(o["dataset"], o["ver"], day, ctx.vintage, f"{o['dataset']}-{ymd}.parquet"),
                 data, "application/vnd.apache.parquet")
            if not ok:
                res.failed = True
                res.reasons.append(f"decode-back failed: {o['dataset']}")
        elif o["kind"] == "tile" and k in cubes:
            cube = cubes[k]
            cube.step_labels = step_labels
            _finish_tile(ctx, res, out, cube, day, grid, supersedes)
            for ov in o.get("overviews", []):
                ocube = cube.overview(int(ov["factor"]), ctx.staging)
                oe = DatasetEntry(ov["layer"], o["ver"], ctx.vintage)
                oe["era"], oe["supersedes"] = list(out.entry["era"]), supersedes
                oe["sources"] = list(out.entry["sources"])
                oe["steps"] = {"expected": out.entry["steps"]["expected"], "got": out.entry["steps"]["got"],
                               "missing": list(out.entry["steps"]["missing"])}
                res.entries.append(oe)
                ov_out = _Out({**o, "layer": ov["layer"]}, oe, out.steps_used)
                _finish_tile(ctx, res, ov_out, ocube, day, None, supersedes, layer=ov["layer"])
                ocube.close()
            cube.close()
        elif o["kind"] == "boxflags" and out.daymax is not None and grid is not None:
            factor = sparse.MPH_PER_MS if o["rule"] == "gust_mph" else 1.0
            lats, lons = grid.latlons()
            boxes = stormbox.grid_rule(out.daymax * factor, lats, lons, o["threshold"], o["min_cells"])
            doc = {"schema": "box-flags/1", "day": day.isoformat(), "rule": o["rule"], "threshold": o["threshold"],
                   "min_cells": o["min_cells"], "boxes": boxes}
            _put(ctx, res, out.entry, src_key(o["dataset"], o["ver"], day, ctx.vintage, f"{o['dataset']}-{ymd}.json"),
                 canonical_json(doc), "application/json")


def _process_grib(ctx, group, day, outputs, supersedes, res):
    steps = day_steps(day, group["cadence_min"])
    fields = sorted({o["field"] for o in outputs})
    off = _dt.timedelta(minutes=int(group.get("valid_offset_min", 0)))
    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        fetched = list(pool.map(lambda w: fetch_grib_step(ctx.http, group, fields, w), steps))
    outs = _init_outputs(ctx, group, day, outputs, len(steps), None, supersedes, res)
    cubes, grid, shape = {}, None, None
    for i, when in enumerate(steps):
        valid = iso(when + off)
        for fname in fields:
            r, st, tries, fb = fetched[i][fname]
            if r is None:
                for out in outs:
                    if out.spec["field"] == fname and i in out.steps_used:
                        out.entry.add_missing(valid, st, tries)
                continue
            src = r.source_record(valid_utc=valid, fallback=fb)
            for out in outs:
                if out.spec["field"] == fname and i in out.steps_used:
                    out.entry.add_source(**src)
            fld = ctx.decode(r.body, group["fields"][fname], shape)
            if shape is None:
                shape, grid = fld.values.shape, fld.grid
            _feed(ctx, outs, fname, i, fld.values, grid, len(steps), cubes)
            fetched[i][fname] = None  # release the bytes
    labels = [iso(w + off) for w in steps]
    _finalize(ctx, group, day, outs, cubes, grid, supersedes, res, labels)
    if group.get("gap_rule"):
        for out in outs:
            if ctx.breaker.gap_failed(out.entry["steps"]["missing"]):
                res.failed = True
                res.reasons.append(f"gaps: {output_id(out.spec)}")


def _get_file(ctx, group, product, when):
    """(Response, fallback flag) trying the configured sources in order; (None, status, tries) if none has it."""
    p = group["products"][product]
    last = (404, 1)
    for src in group["source_order"]:
        if src == "aws":
            url = nodd_url(group["bucket"], fill(p["aws"], when))
        else:
            url = fill(p[src], when)
        r = ctx.http.fetch(url)
        if r.ok:
            return r, (None if src == group["source_order"][0] else src), None
        last = (r.status, r.tries)
    return None, None, last


def _process_files(ctx, group, day, outputs, supersedes, res):
    steps = day_steps(day, group["cadence_min"])
    products = sorted({p for o in outputs for p in (o.get("products") or [o.get("product")]) if p})
    outs = _init_outputs(ctx, group, day, outputs, len(steps), None, supersedes, res)
    for out in outs:
        if out.spec["kind"] == "tar":
            out.entry["steps"]["expected"] = len(steps) * len(out.spec["products"])
    members, cubes, grid, shape = [], {}, None, None
    jobs = [(i, w, p) for i, w in enumerate(steps) for p in products]
    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        got = list(pool.map(lambda j: _get_file(ctx, group, j[2], j[1]), jobs))
    for (i, when, prod), (r, fb, miss) in zip(jobs, got):
        valid = iso(when)
        tar_outs = [o for o in outs if o.spec["kind"] == "tar" and prod in o.spec["products"]]
        other = [o for o in outs if o.spec.get("product") == prod]
        if r is None:
            for o in tar_outs + other:
                o.entry.add_missing(valid, miss[0], miss[1])
            continue
        src = r.source_record(valid_utc=valid, fallback=fb)
        for o in tar_outs + other:
            o.entry.add_source(**src)
        name = f"{prod}/{r.url.rsplit('/', 1)[1]}"
        members.append((name, r.body))
        for o in tar_outs:
            o.entry["steps"]["got"] += 1
        if other:
            fld = ctx.decode(gzip.decompress(r.body), {"grib": None}, shape)
            if shape is None:
                shape, grid = fld.values.shape, fld.grid
            _feed(ctx, other, prod, i, fld.values, grid, len(steps), cubes)
    for out in outs:
        if out.spec["kind"] == "tar" and members:
            mine = [m for m in members if m[0].split("/")[0] in out.spec["products"]]
            data = compact.tar_bytes(mine)
            back = compact.untar(data)
            ok = compact.member_records(back) == compact.member_records(mine)
            out.entry["checks"].update({"roundtrip": "pass" if ok else "fail", "members": compact.member_records(mine)})
            _put(ctx, res, out.entry, src_key(out.spec["dataset"], out.spec["ver"], day, ctx.vintage,
                                              f"{out.spec['dataset']}-{day.strftime('%Y%m%d')}.tar"), data,
                 "application/x-tar")
            if not ok:
                res.failed = True
    _finalize(ctx, group, day, [o for o in outs if o.spec["kind"] != "tar"], cubes, grid, supersedes, res,
              [iso(w) for w in steps])


def _listed_times(ctx, group, day):
    keys = list_keys(ctx.http, group["bucket"], fill(group["prefix"], day_steps(day, 1440)[0]))
    rx = re.compile(group["time_regex"])
    out = []
    for k in keys:
        m = rx.search(k["key"])
        if m:
            t = _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=_dt.timezone.utc)
            out.append((t, k))
    out.sort(key=lambda x: x[0])
    return out


def _process_listed(ctx, group, day, outputs, supersedes, res):
    steps = day_steps(day, group["cadence_min"])
    listed = _listed_times(ctx, group, day)
    win = _dt.timedelta(minutes=int(group.get("pick_window_min", 2)))
    picks = []
    for w in steps:
        cand = [k for t, k in listed if w <= t < w + win]
        picks.append(cand[0] if cand else None)
    outs = _init_outputs(ctx, group, day, outputs, len(steps), None, supersedes, res)
    for o in outs:
        o.spec = {**o.spec, "field": "_"}

    def get(k):
        return None if k is None else ctx.http.fetch(nodd_url(group["bucket"], k["key"]))

    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        resp = list(pool.map(get, picks))
    cubes, grid, shape = {}, None, None
    for i, (w, r) in enumerate(zip(steps, resp)):
        valid = iso(w)
        if r is None or not r.ok:
            for o in outs:
                o.entry.add_missing(valid, 404 if r is None else r.status, 0 if r is None else r.tries)
            continue
        for o in outs:
            o.entry.add_source(**r.source_record(valid_utc=valid))
        fld = ctx.decode(gzip.decompress(r.body), {"grib": None}, shape)
        if shape is None:
            shape, grid = fld.values.shape, fld.grid
        _feed(ctx, outs, "_", i, fld.values, grid, len(steps), cubes)
        resp[i] = None
    _finalize(ctx, group, day, outs, cubes, grid, supersedes, res, [iso(w) for w in steps])


def _process_listed_all(ctx, group, day, outputs, supersedes, res):
    listed = _listed_times(ctx, group, day)
    outs = _init_outputs(ctx, group, day, outputs, 0, None, supersedes, res)
    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        resp = list(pool.map(lambda tk: ctx.http.fetch(nodd_url(group["bucket"], tk[1]["key"])), listed))
    members = []
    for o in outs:
        o.entry["steps"]["expected"] = len(listed)
    for (t, k), r in zip(listed, resp):
        if not r.ok:
            for o in outs:
                o.entry.add_missing(iso(t), r.status, r.tries)
            continue
        for o in outs:
            o.entry.add_source(**r.source_record(valid_utc=iso(t)))
            o.entry["steps"]["got"] += 1
        members.append((k["key"].rsplit("/", 1)[1], r.body))
    for o in outs:
        if not members:
            continue
        raw = compact.tar_bytes(members)
        data = compact.zstd(raw)
        back = compact.untar(compact.unzstd(data))
        ok = compact.member_records(back) == compact.member_records(members)
        o.entry["checks"].update({"roundtrip": "pass" if ok else "fail", "members": len(members)})
        _put(ctx, res, o.entry, src_key(o.spec["dataset"], o.spec["ver"], day, ctx.vintage,
                                        f"{o.spec['dataset']}-{day.strftime('%Y%m%d')}.tar.zst"), data,
             "application/zstd")
        if not ok:
            res.failed = True


# ---------------------------------------------------------------- revisions

def source_etags_changed(ctx, group, day, entries):
    """True if a NODD object recorded in ``entries`` changed its ETag, or a step that was missing now exists."""
    known = {}
    for e in entries:
        for s in e.get("sources", []):
            if s.get("etag") and s["url"].startswith(f"https://{group['bucket']}.s3.amazonaws.com/"):
                known[urllib.parse.unquote(s["url"].split(".s3.amazonaws.com/", 1)[1])] = s["etag"]
    if group["kind"] == "grib":
        prefix = fill(group["key"], day_steps(day, 1440)[0]).rsplit("/", 1)[0] + "/"
    elif group["kind"] in ("listed", "listed-all"):
        prefix = fill(group["prefix"], day_steps(day, 1440)[0])
    else:
        return False
    listing = {k["key"]: k["etag"] for k in list_keys(ctx.http, group["bucket"], prefix)}
    for key, etag in known.items():
        if key in listing and listing[key] != etag:
            log("etag-changed", group["id"], day.isoformat())
            return True
    missing = sum(len(e["steps"]["missing"]) for e in entries)
    if missing and group["kind"] == "grib":
        for e in entries:
            for m in e["steps"]["missing"]:
                when = _dt.datetime.strptime(m["valid_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_dt.timezone.utc)
                when -= _dt.timedelta(minutes=int(group.get("valid_offset_min", 0)))
                if fill(group["key"], when) in listing:
                    log("late-file", group["id"], day.isoformat())
                    return True
    return False
