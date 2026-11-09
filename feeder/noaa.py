"""NOAA Open Data Dissemination (AWS) sources: .idx lookup, GRIB2 header walk when there is no .idx, byte-range
reads, S3 listing, ETag / Last-Modified capture; and the per-day processing of one source group into exact grids
(every cell, source precision), sparse cubes, as-served bundles, box flags and (optional, off by default) u8 display
tiles."""

import concurrent.futures as cf
import copy
import datetime as _dt
import gzip
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np

from . import canonical_json, compact, exact, grib, log, par, sparse, stormbox, tile, utcnow
from .net import HttpError
from .manifest import DatasetEntry, iso
from .store import src_key, tile_key

NODD_WORKERS = 8
PREFETCH_STEPS = 2 * NODD_WORKERS  # downloaded steps waiting for the encoder, at most (bounds memory)


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

def active_outputs(group):
    """The group's outputs that are switched on (``enabled`` defaults to true; display tiles default to off)."""
    return [o for o in group["outputs"] if o.get("enabled", True)]


def apply_cuts(group, cuts):
    g = copy.deepcopy(group)
    g["outputs"] = active_outputs(g)
    for c in cuts:
        if c.get("group") != g["id"]:
            continue
        layers = c.get("layers")
        for o in g["outputs"]:
            if layers is None or output_id(o) in layers:
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


def _put(ctx, res, entry, key, data, ctype, covers=None, exact=None):
    pr = ctx.store.put(key, data, ctype)
    entry.add_object(pr, covers, exact)
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
    par.log_memory(group["id"], day.isoformat())
    for e in res.entries:
        e.finish_status()
        if e["checks"].get("decode_back") == "fail" or e["checks"].get("roundtrip") == "fail":
            e["status"] = "partial"
        if e["checks"].get("station_points") == "unavailable":
            # station cells were wanted but the station catalogue was not readable: never "complete"
            e["status"] = "partial" if e["status"] == "complete" else e["status"]
            res.failed = True
            res.reasons.append(f"station points unavailable: {e['id']}")
    return res


def _new_entry(ctx, o, group, day, supersedes):
    e = DatasetEntry(output_id(o), o["ver"], ctx.vintage)
    e["era"] = _era(group, day)
    e["supersedes"] = supersedes
    return e


def station_map(grid, points, window=6):
    """[[station id, nearest cell, degree cell]] for ``points`` [(id, lat, lon)].

    nearest = the grid point nearest in the grid's own projection; degree cell = the minimum of
    (lat - lat0)**2 + (lon - lon0)**2 over the grid's float64 latitude/longitude arrays (a common lookup that can
    differ from the projected nearest point), searched around the nearest point and widened while the minimum sits
    on the search window's edge.
    Stations outside the grid get [id, None, None]."""
    lats, lons = grid.latlons()
    ny, nx = grid.shape
    out = []
    for sid, lat, lon in points:
        c = grid.locate(lat, lon)
        if c is None:
            out.append([sid, None, None])
            continue
        j, i = divmod(int(c), nx)
        w = window
        while True:
            j0, j1, i0, i1 = max(0, j - w), min(ny, j + w + 1), max(0, i - w), min(nx, i + w + 1)
            d = (lats[j0:j1, i0:i1] - lat) ** 2 + (lons[j0:j1, i0:i1] - lon) ** 2
            k = int(np.argmin(d))
            jj, ii = divmod(k, i1 - i0)
            edge = (jj == 0 and j0 > 0) or (jj == j1 - j0 - 1 and j1 < ny) or (ii == 0 and i0 > 0) or \
                (ii == i1 - i0 - 1 and i1 < nx)
            if not edge or (j0 == 0 and i0 == 0 and j1 == ny and i1 == nx):
                break
            w *= 4
        out.append([sid, int(c), (j0 + jj) * nx + (i0 + ii)])
    return out


def halo_cells(cells, shape, r=1):
    """Every cell within ``r`` rows/columns of each cell (a (2r+1) x (2r+1) block, clipped at the grid edge)."""
    ny, nx = shape
    out = set()
    for c in cells:
        if c is None:
            continue
        j, i = divmod(int(c), nx)
        for dj in range(-r, r + 1):
            for di in range(-r, r + 1):
                if 0 <= j + dj < ny and 0 <= i + di < nx:
                    out.add((j + dj) * nx + i + di)
    return sorted(out)


def _station_map(ctx, grid, cache):
    """The day's station map for ``grid`` (computed once per group), or None without a station catalogue."""
    if "map" not in cache:
        pts = ctx.station_points()
        cache["map"] = None if pts is None else station_map(grid, pts)
    return cache["map"]


def _init_outputs(ctx, group, day, outputs, n_steps, grid, supersedes, res):
    outs = []
    for o in outputs:
        e = _new_entry(ctx, o, group, day, supersedes)
        if o.get("station_points"):
            e["station_map"] = None
        res.entries.append(e)
        every = int(o.get("every_n_steps", 1))
        used = [i for i in range(n_steps) if i % every == 0]
        e["steps"]["expected"] = len(used)
        outs.append(_Out(o, e, set(used)))
    return outs


def _attach_station_map(ctx, out, grid, smap_cache):
    smap = _station_map(ctx, grid, smap_cache)
    if smap is None:
        out.entry["checks"]["station_points"] = "unavailable"
        return None
    out.entry["checks"]["station_points"] = len([m for m in smap if m[1] is not None])
    if "on" in smap_cache:
        # the group's first station output carries the map; the others point to it (same grid, same catalogue)
        out.entry["station_map"] = {"same_as": smap_cache["on"]}
    else:
        smap_cache["on"] = output_id(out.spec)
        out.entry["station_map"] = {"catalog": ctx.station_catalog_id(), "rule": "nearest = projected grid point; "
                                    "degree = min (lat-lat0)^2 + (lon-lon0)^2 over the float64 grid lat/lon",
                                    "stations": smap}
    return smap


def _exact_params(ctx, o):
    """The exact-grid cube parameters of output ``o`` (without the grid shape)."""
    xc = ctx.cfg.get("exact", {})
    return {"tile_size": int(o.get("tile_size", xc.get("tile_size", 128))), "predictor": o.get("predictor", "dxy"),
            "level": int(o.get("zstd_level", xc.get("zstd_level", 19))), "store": o.get("store", "auto")}


def _feed(ctx, outs, field_key, i, fld, grid, n_steps, cubes, label=None, source=None, source_encoding="none",
          smap_cache=None):
    """``fld``: a decoded grib.Field, or a StepOut from ``step_job`` (exact steps already prepared, keyed by the
    output's position in ``outs``; values present when another output needs them)."""
    values = fld.values
    shape = tuple(getattr(fld, "shape", None) or values.shape)
    prepared = getattr(fld, "prepared", None) or {}
    smap_cache = {} if smap_cache is None else smap_cache
    for n, out in enumerate(outs):
        o = out.spec
        if (o.get("field") or o.get("product")) != field_key:
            continue
        if i not in out.steps_used:
            continue
        k = id(out)
        if o["kind"] == "exact":
            if k not in cubes:
                xp = _exact_params(ctx, o)
                cubes[k] = exact.ExactCube(shape, n_steps, ctx.staging, xp["tile_size"], xp["predictor"], xp["level"],
                                           xp["store"], int(ctx.cfg.get("exact", {}).get("workers", 4)))
                if o.get("station_points"):
                    _attach_station_map(ctx, out, grid, smap_cache)
            if n in prepared:
                cubes[k].commit_step(i, label, prepared[n])
            else:
                cubes[k].add_step(i, label, values, fld.packing, source, source_encoding)
        elif o["kind"] == "sparse":
            if k not in cubes:
                points, deg = None, None
                if o.get("station_points"):
                    smap = _attach_station_map(ctx, out, grid, smap_cache)
                    if smap is not None:
                        points = halo_cells([m[1] for m in smap], shape, int(o.get("halo", 1)))
                        deg = [m[2] for m in smap]
                cubes[k] = sparse.SparseCube(shape, n_steps, o["threshold_mph"] / sparse.MPH_PER_MS,
                                             o.get("lattice_step", 10), points, deg, exact=o.get("exact", True))
            cubes[k].add_step(i, values, fld.packing)
        elif o["kind"] == "tile":
            if k not in cubes:
                enc = tile.Encoding(ctx.cfg["encodings"][o["enc"]])
                cubes[k] = tile.TileCube(shape, n_steps, enc, ctx.staging, ctx.cfg.get("tile_size", 128))
            cubes[k].add_step(i, values)
        elif o["kind"] == "boxflags":
            out.daymax = values.copy() if out.daymax is None else np.fmax(out.daymax, values)
        out.entry["steps"]["got"] += 1


def _finalize(ctx, group, day, outs, cubes, grid, supersedes, res, step_labels):
    ymd = day.strftime("%Y%m%d")
    for out in outs:
        o = out.spec
        k = id(out)
        if o["kind"] == "exact" and k in cubes:
            cube = cubes[k]
            for m in out.entry["steps"]["missing"]:
                if m["valid_utc"] in step_labels:
                    cube.add_missing(step_labels.index(m["valid_utc"]), m["valid_utc"])
            meta = {"dataset": o["dataset"], "day": day.isoformat(), "source_group": group["id"],
                    "field": o.get("field") or o.get("product"),
                    "grid": list(grid.signature) if grid is not None and grid.signature else None,
                    "step_labels": step_labels}
            data = cube.to_bytes(meta)
            hdr = exact.ExactReader.from_bytes(data).header
            field = group.get("fields", {}).get(o.get("field"), {"grib": None})
            ok, rep = exact.verify(data, ctx.decode, {"grib": field.get("grib")}, **_verify_par(ctx))
            out.entry["checks"].update({"decode_back": "pass" if ok else "fail", "exact": cube.summary(),
                                        "cells_checked": rep["checked"] * cube.shape[0] * cube.shape[1]})
            _put(ctx, res, out.entry, src_key(o["dataset"], o["ver"], day, ctx.vintage, f"{o['dataset']}-{ymd}.xgrid"),
                 data, "application/octet-stream", exact=exact.manifest_record(hdr))
            cube.close()
            if not ok:
                res.failed = True
                res.reasons.append(f"exact decode-back failed: {o['dataset']} steps={rep['failed'][:5]}")
        elif o["kind"] == "sparse" and k in cubes:
            cube = cubes[k]
            data, rows = cube.to_parquet({"dataset": o["dataset"], "day": day.isoformat(), "field": o["field"],
                                          "step_labels": step_labels, "unit": "m/s"})
            ok = sparse.decode_check(data, cube)
            out.entry["checks"].update({"decode_back": "pass" if ok else "fail", "rows": rows, **cube.checks()})
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


# ---------------------------------------------------------------- per-step work (worker processes)

class StepOut:
    """What the main loop needs from one decoded step: the values (only when an output other than an exact grid
    reads them), the packing, the grid shape, and each exact output's prepared step (exact.prepare_step)."""

    def __init__(self, values, packing, shape, prepared, field=None):
        self.values = values
        self.packing = packing
        self.shape = tuple(shape)
        self.prepared = prepared
        self.field = field  # the decoded grib.Field itself, inline only (its grid cannot cross processes)


def step_job(decode, raw, gz, field, exact_params, want_values, threads, inline):
    """Decode one fetched message and prepare its exact-grid steps; runs in a worker process (or inline)."""
    fld = decode(gzip.decompress(raw) if gz else raw, field, None)
    shape = fld.values.shape
    prepared = {n: exact.prepare_step(fld.values, fld.packing, raw, "gzip" if gz else "none", shape=shape,
                                      workers=threads, **xp) for n, xp in exact_params.items()}
    return StepOut(fld.values if (want_values or inline) else None, fld.packing, shape, prepared,
                   fld if inline else None)


class _Steps:
    """Starts ``step_job`` for fetched steps (in worker processes when there are any) and tracks the group's grid:
    the first decoded step sets the shape and grid that every later step must match, as in the serial loop."""

    def __init__(self, ctx, outs):
        self.ctx = ctx
        self.outs = outs
        self.remote = par.pool() is not None and par.picklable(ctx.decode)
        self.threads = 1 if self.remote else int(ctx.cfg.get("exact", {}).get("workers", 4))
        self.ahead = par.width() if self.remote else 1  # jobs in flight (inline work runs one step at a time)
        self.shape = None
        self.grid = None

    def start(self, i, field_key, raw, gz, field):
        exact_params, want = {}, False
        for n, out in enumerate(self.outs):
            o = out.spec
            if (o.get("field") or o.get("product")) != field_key or i not in out.steps_used:
                continue
            if o["kind"] == "exact":
                exact_params[n] = _exact_params(self.ctx, o)
            else:
                want = True
        args = (self.ctx.decode, raw, gz, field, exact_params, want, self.threads, not self.remote)
        return par.pool().submit(step_job, *args) if self.remote else par.inline(step_job, *args)

    def accept(self, step, raw, gz, field):
        """Checks the step against the group's grid (the first one sets it)."""
        if self.shape is None:
            self.shape = step.shape
            if step.field is not None:
                self.grid = step.field.grid
            else:  # the grid object stays in the main process: decode the first message here for it
                self.grid = self.ctx.decode(gzip.decompress(raw) if gz else raw, field, None).grid
        elif step.shape != self.shape:
            raise grib.GribError(f"grid shape {step.shape} != expected {self.shape}")


def _verify_par(ctx):
    """exact.verify arguments that spread the decode-back over the worker processes (none without them)."""
    if par.pool() is None or not par.picklable(ctx.decode):
        return {}
    return {"submit": par.submit, "parts": 2 * par.procs(), "staging_dir": ctx.staging}


def _process_grib(ctx, group, day, outputs, supersedes, res):
    steps = day_steps(day, group["cadence_min"])
    fields = sorted({o["field"] for o in outputs})
    off = _dt.timedelta(minutes=int(group.get("valid_offset_min", 0)))
    outs = _init_outputs(ctx, group, day, outputs, len(steps), None, supersedes, res)
    known = set(group.get("known_gaps", []))
    strict = bool(group.get("gap_rule"))
    cubes, smap = {}, {}
    work = _Steps(ctx, outs)

    def start(item):
        i, fname, (r, st, tries, fb) = item
        if r is None:
            return None
        return work.start(i, fname, r.body, False, group["fields"][fname])

    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        fetched = par.prefetch(pool, lambda w: fetch_grib_step(ctx.http, group, fields, w), steps, PREFETCH_STEPS)
        items = ((i, fname, got[fname]) for i, got in enumerate(fetched) for fname in fields)
        for (i, fname, (r, st, tries, fb)), step in par.ordered(start, items, work.ahead):
            when = steps[i]
            valid = iso(when + off)
            if r is None:
                for out in outs:
                    if out.spec["field"] == fname and i in out.steps_used:
                        out.entry.add_missing(valid, st, tries, known_gap=(iso(when) in known) if strict else None)
                continue
            src = r.source_record(valid_utc=valid, fallback=fb)
            for out in outs:
                if out.spec["field"] == fname and i in out.steps_used:
                    out.entry.add_source(**src)
            work.accept(step, r.body, False, group["fields"][fname])
            _feed(ctx, outs, fname, i, step, work.grid, len(steps), cubes, label=valid, source=r.body,
                  smap_cache=smap)
    labels = [iso(w + off) for w in steps]
    _finalize(ctx, group, day, outs, cubes, work.grid, supersedes, res, labels)
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
            _feed(ctx, other, prod, i, fld, grid, len(steps), cubes, label=valid, source=r.body,
                  source_encoding="gzip")
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


def _past_day(day):
    return day < utcnow().date()


def _process_listed(ctx, group, day, outputs, supersedes, res):
    steps = day_steps(day, group["cadence_min"])
    listed = _listed_times(ctx, group, day)
    empty = not listed and _past_day(day)
    win = _dt.timedelta(minutes=int(group.get("pick_window_min", 2)))
    picks = []
    for w in steps:
        cand = [k for t, k in listed if w <= t < w + win]
        picks.append(cand[0] if cand else None)
    outs = _init_outputs(ctx, group, day, outputs, len(steps), None, supersedes, res)
    for o in outs:
        o.spec = {**o.spec, "field": "_"}
    known = set(group.get("known_gaps", []))
    strict = bool(group.get("gap_rule"))

    def get(k):
        return None if k is None else ctx.http.fetch(nodd_url(group["bucket"], k["key"]))

    cubes = {}
    work = _Steps(ctx, outs)

    def start(item):
        i, r = item
        return None if r is None or not r.ok else work.start(i, "_", r.body, True, {"grib": None})

    with cf.ThreadPoolExecutor(max_workers=NODD_WORKERS) as pool:
        fetched = par.prefetch(pool, get, picks, PREFETCH_STEPS)
        for (i, r), step in par.ordered(start, enumerate(fetched), work.ahead):
            valid = iso(steps[i])
            if r is None or not r.ok:
                for o in outs:
                    # an empty listing for a past day is not a source gap: status 0, never "complete"
                    o.entry.add_missing(valid, 0 if empty else (404 if r is None else r.status),
                                        0 if r is None else r.tries,
                                        known_gap=(valid in known) if strict else None)
                continue
            for o in outs:
                o.entry.add_source(**r.source_record(valid_utc=valid))
            work.accept(step, r.body, True, {"grib": None})
            _feed(ctx, outs, "_", i, step, work.grid, len(steps), cubes, label=valid, source=r.body,
                  source_encoding="gzip")
    grid = work.grid
    _finalize(ctx, group, day, outs, cubes, grid, supersedes, res, [iso(w) for w in steps])
    if strict:
        for o in outs:
            if ctx.breaker.gap_failed(o.entry["steps"]["missing"]):
                res.failed = True
                res.reasons.append(f"gaps: {output_id(o.spec)}")
    if empty:
        res.failed = True
        res.reasons.append(f"empty listing: {group['id']}")


def _process_listed_all(ctx, group, day, outputs, supersedes, res):
    listed = _listed_times(ctx, group, day)
    outs = _init_outputs(ctx, group, day, outputs, 0, None, supersedes, res)
    if not listed and _past_day(day):
        # nothing listed for a past day: the day is missing, not complete
        for o in outs:
            o.entry["steps"]["expected"] = 1
            o.entry.add_missing(iso(day_steps(day, 1440)[0]), 0, 1)
        res.failed = True
        res.reasons.append(f"empty listing: {group['id']}")
        return
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
