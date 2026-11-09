"""Dry-run plan for one day: ``python -m feeder.dryplan`` (uploads nothing, needs no storage credentials).

Reads only what sizes a day: the .idx file of every gridded step (byte sizes of the wanted messages) and the day
listings of the file sources. With SAMPLE=1 it also fetches ONE step of every exact layer by byte range (a single
GRIB2 message, or one served file), builds the exact-grid object, runs the full decode-back check against the
decoder's values, and measures the stored / source ratio; the sparse cubes are built on the same step (with the
station map when STATIONS=1, which reads the station catalogues). The day's stored bytes are projected from those
ratios, and the total to date from the day.

Environment: IN_DAY (YYYY-MM-DD), SAMPLE (1 / 0, default 1), SAMPLE_HOUR (UTC hour, default 21), STATIONS (1 / 0,
default 1), CAP_GB (default 2500), OTHER_LANES_GB (observations and warnings, default 310).
"""

import concurrent.futures as cf
import datetime as _dt
import gzip
import os
import sys
import tempfile
import time

import numpy as np

from . import canonical_json, compact, exact, grib, iem, load_config, log, noaa, sparse, tile, utcnow
from .net import Http
from .plan import ARCHIVE_START, PlanError, parse_day

TAR_MEMBER_OVERHEAD = 1536  # PAX header blocks + padding per member, upper bound


class _Ctx:
    """The parts of run.Ctx the sampling needs (station catalogue read live)."""

    def __init__(self, cfg, http):
        self.cfg, self.http = cfg, http


def _idx_sizes(http, group, fields, when):
    """{field: message bytes or None}, url, idx bytes for one step."""
    url = noaa.nodd_url(group["bucket"], noaa.fill(group["key"], when))
    r = http.fetch(url + group["idx_suffix"])
    if not r.ok:
        return {f: None for f in fields}, url, 0, r.status
    entries = noaa.parse_idx(r.body.decode("ascii", "replace"))
    out = {}
    for f in fields:
        e = noaa.find_entry(entries, group["fields"][f]["idx"])
        if e is None:
            out[f] = None
        elif e.end is not None:
            out[f] = e.end - e.offset + 1
        else:
            h = http.fetch(url, rng=(0, 0))
            total = int(h.headers.get("content-range", "/0").rsplit("/", 1)[1] or 0)
            out[f] = total - e.offset
        out[f + "@"] = e
    return out, url, len(r.body), 200


def _sample_grib(http, group, field, when, entry, url):
    r = http.get(url, rng=(entry.offset, entry.end))
    return r.body


def _exact_sample(cfg, o, msg, source, encoding, field_ids, staging, sample_cells=None):
    """Builds a one-step exact object; returns measurements and the decoded field."""
    t0 = time.monotonic()
    fld = grib.decode(msg, {"grib": field_ids})
    xc = cfg.get("exact", {})
    cube = exact.ExactCube(fld.values.shape, 1, staging, int(xc.get("tile_size", 128)), o.get("predictor", "dxy"),
                           int(xc.get("zstd_level", 19)), o.get("store", "auto"), int(xc.get("workers", 4)),
                           o.get("palette", "off"))
    rec = cube.add_step(0, "sample", fld.values, fld.packing, source, encoding)
    data = cube.to_bytes({"dataset": o["dataset"], "sample": True})
    t1 = time.monotonic()
    ok, rep = exact.verify(data, grib.decode, {"grib": field_ids})
    t2 = time.monotonic()
    cube.close()
    meas = {"source_bytes": len(source), "stored_bytes": len(data), "mode": rec["mode"],
            "ints_bytes": rec.get("ints_bytes"), "int_exact": rec["int_exact"],
            "packing": rec["packing"], "decode_back": "pass" if ok else "fail", "cells": int(fld.values.size),
            "encode_s": round(t1 - t0, 2), "verify_s": round(t2 - t1, 2),
            "ratio": len(data) / len(source)}
    return meas, fld


def _sparse_sample(o, fld, smap):
    points = deg = None
    if smap is not None:
        points = noaa.halo_cells([m[1] for m in smap], fld.values.shape, int(o.get("halo", 1)))
        deg = [m[2] for m in smap]
    cube = sparse.SparseCube(fld.values.shape, 1, o["threshold_mph"] / sparse.MPH_PER_MS, o.get("lattice_step", 10),
                             points, deg, exact=o.get("exact", True))
    cube.add_step(0, fld.values, fld.packing)
    data, rows = cube.to_parquet({"dataset": o["dataset"], "sample": True})
    ok = sparse.decode_check(data, cube)
    return {"bytes_per_step": len(data), "rows": rows, "decode_back": "pass" if ok else "fail", **cube.checks(),
            "station_points": None if smap is None else len([m for m in smap if m[1] is not None]),
            "degree_cell_differs": None if smap is None else
            len([m for m in smap if m[1] is not None and m[1] != m[2]])}


def plan_day(cfg, http, day, sample=True, sample_hour=21, stations=True):
    out = {"day": day.isoformat(), "groups": {}, "outputs": {}, "iem_requests": {}}
    staging = tempfile.mkdtemp(prefix="dryplan-")
    smap_by_grid = {}
    points = iem.station_points_live(_Ctx(cfg, http)) if (sample and stations) else None
    try:
        for g in [noaa.apply_cuts(g, []) for g in cfg["noaa"]["groups"]]:
            outs = g["outputs"]
            steps = noaa.day_steps(day, g.get("cadence_min", 1440))
            gi = {"kind": g["kind"], "steps": len(steps)}
            src_by_key = {}
            if g["kind"] == "grib":
                fields = sorted({o["field"] for o in outs})
                with cf.ThreadPoolExecutor(max_workers=noaa.NODD_WORKERS) as pool:
                    res = list(pool.map(lambda w: _idx_sizes(http, g, fields, w), steps))
                gi["idx_bytes"] = sum(r[2] for r in res)
                gi["idx_missing"] = sum(1 for r in res if r[3] != 200)
                for f in fields:
                    sizes = [r[0][f] for r in res if r[0].get(f)]
                    src_by_key[f] = {"steps_found": len(sizes), "bytes": sum(sizes)}
                k = min(range(len(steps)), key=lambda i: abs(steps[i].hour - sample_hour) + steps[i].minute / 60)
                sample_step = (steps[k], res[k])
            elif g["kind"] == "files":
                for prod, p in g["products"].items():
                    prefix = noaa.fill(p["aws"], steps[0]).rsplit("/", 1)[0] + "/"
                    keys = {x["key"]: x["bytes"] for x in noaa.list_keys(http, g["bucket"], prefix)}
                    want = [noaa.fill(p["aws"], w) for w in steps]
                    got = [keys[w] for w in want if w in keys]
                    src_by_key[prod] = {"steps_found": len(got), "bytes": sum(got), "sample_key": want[sample_hour]
                                        if want[sample_hour] in keys else None}
            elif g["kind"] == "listed":
                listed = noaa._listed_times(_Ctx(cfg, http), g, day)
                win = _dt.timedelta(minutes=int(g.get("pick_window_min", 2)))
                picks = []
                for w in steps:
                    c = [kk for t, kk in listed if w <= t < w + win]
                    picks.append(c[0] if c else None)
                got = [p for p in picks if p]
                src_by_key["_"] = {"steps_found": len(got), "bytes": sum(p["bytes"] for p in got), "listed": len(listed)}
                k = min(range(len(steps)), key=lambda i: abs(steps[i].hour - sample_hour) + steps[i].minute / 60)
                sample_step = picks[k] or next((p for p in picks if p), None)
            elif g["kind"] == "listed-all":
                listed = noaa._listed_times(_Ctx(cfg, http), g, day)
                src_by_key["_"] = {"steps_found": len(listed), "bytes": sum(kk["bytes"] for _, kk in listed)}
                gi["steps"] = len(listed)
                sample_step = listed
            gi["sources"] = src_by_key
            out["groups"][g["id"]] = gi
            for o in outs:
                oid = noaa.output_id(o)
                key = o.get("field") or o.get("product") or "_"
                src = src_by_key.get(key, {"steps_found": 0, "bytes": 0})
                rec = {"kind": o["kind"], "ver": o["ver"], "objects": 1, "steps_expected": gi["steps"],
                       "steps_found": src.get("steps_found"), "source_bytes": src.get("bytes")}
                if o["kind"] == "tar":
                    prods = o.get("products") or ["_"]
                    nb = sum(src_by_key[p]["bytes"] for p in prods)
                    nm = sum(src_by_key[p]["steps_found"] for p in prods)
                    rec.update(source_bytes=nb, steps_found=nm, steps_expected=gi["steps"] * len(prods)
                               if g["kind"] == "files" else gi["steps"])
                    rec["stored_bytes"] = nb + nm * TAR_MEMBER_OVERHEAD
                    if g["kind"] == "listed-all" and sample and sample_step:
                        picks = [sample_step[i][1] for i in sorted({0, len(sample_step) // 2, len(sample_step) - 1})]
                        bodies = [http.get(noaa.nodd_url(g["bucket"], p["key"])).body for p in picks]
                        raw = compact.tar_bytes([(p["key"].rsplit("/", 1)[1], b) for p, b in zip(picks, bodies)])
                        ratio = len(compact.zstd(raw)) / len(raw)
                        rec["sample"] = {"files": len(picks), "raw_bytes": len(raw), "zstd_ratio": round(ratio, 4)}
                        rec["stored_bytes"] = int(rec["stored_bytes"] * ratio)
                elif o["kind"] == "boxflags":
                    rec["stored_bytes"] = 2048
                elif o["kind"] == "exact" and sample:
                    if g["kind"] == "grib":
                        when, (sizes, url, _, st) = sample_step
                        e = sizes.get(o["field"] + "@")
                        body = _sample_grib(http, g, o["field"], when, e, url)
                        meas, fld = _exact_sample(cfg, o, body, body, "none", g["fields"][o["field"]]["grib"], staging)
                        meas["valid_utc"] = noaa.iso(when)
                    else:
                        r = http.get(noaa.nodd_url(g["bucket"], sample_step["key"]))
                        meas, fld = _exact_sample(cfg, o, gzip.decompress(r.body), r.body, "gzip", None, staging)
                        meas["key"] = sample_step["key"].rsplit("/", 1)[1]
                    rec["sample"] = meas
                    rec["stored_bytes"] = int((src.get("bytes") or 0) * meas["ratio"])
                    if o.get("station_points") or any(x["kind"] == "sparse" and x.get("field") == o.get("field")
                                                      for x in outs):
                        gkey = g["id"]
                        if gkey not in smap_by_grid:
                            smap_by_grid[gkey] = None if points is None else noaa.station_map(fld.grid, points)
                        for so in outs:
                            if so["kind"] == "sparse" and so["field"] == o["field"]:
                                sp = _sparse_sample(so, fld, smap_by_grid[gkey])
                                out["outputs"][noaa.output_id(so)] = {
                                    "kind": "sparse", "ver": so["ver"], "objects": 1, "steps_expected": gi["steps"],
                                    "steps_found": src.get("steps_found"), "sample": sp,
                                    "stored_bytes": sp["bytes_per_step"] * gi["steps"]}
                    del fld
                elif o["kind"] == "sparse":
                    rec.setdefault("stored_bytes", None)
                if o["kind"] == "sparse" and oid in out["outputs"]:
                    continue
                out["outputs"][oid] = rec
            if g["id"] == "mrms-mesh" and sample:
                key = src_by_key.get("mesh60", {}).get("sample_key")
                if key:
                    r = http.get(noaa.nodd_url(g["bucket"], key))
                    fld = grib.decode(gzip.decompress(r.body), {"grib": None})
                    enc = tile.Encoding(cfg["encodings"]["mesh05"])
                    codes, ov = enc.encode(fld.values)
                    v = fld.values
                    out["groups"][g["id"]]["mesh_check"] = {
                        "file": key.rsplit("/", 1)[1], "cells": int(v.size), "no_coverage_-3": int((v == -3).sum()),
                        "no_hail_-1": int((v == -1).sum()), "hail_below_12.7mm": int(((v > 0) & (v < 12.7)).sum()),
                        "hail_12.7mm_up": int((v >= 12.7).sum()), "mesh05_overflow_cells": int(ov.sum())}
    finally:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
    # observation lane: requests for the day (no download)
    states = cfg["iem"]["states"]
    for ds in cfg["iem"]["datasets"]:
        if not ds.get("enabled"):
            continue
        if ds["kind"] in ("catalog", "table"):
            out["iem_requests"][ds["id"]] = len(iem.networks(cfg["iem"], ds))
        else:
            out["iem_requests"][ds["id"]] = 1
    for ds in cfg["other"]["datasets"]:
        if ds.get("enabled") and ds["kind"] == "daily-files":
            out["iem_requests"][ds["id"]] = len(ds["suffixes"])
    out["iem_requests"]["states"] = len(states)
    return out


def summarise(out, cfg, cap_gb, other_gb, today=None):
    today = today or utcnow().date()
    days = (today - ARCHIVE_START).days
    day_bytes = sum(r.get("stored_bytes") or 0 for r in out["outputs"].values())
    src_bytes = sum(r.get("source_bytes") or 0 for r in out["outputs"].values() if r["kind"] in ("exact", "tar"))
    s = {"days_to_date": days, "objects_per_day": len(out["outputs"]) + 1,
         "stored_bytes_day": day_bytes, "source_bytes_day": src_bytes,
         "projected_grid_gb_to_date": round(day_bytes * days / 1e9, 1),
         "other_lanes_gb": other_gb, "cap_gb": cap_gb}
    s["projected_total_gb_to_date"] = round(s["projected_grid_gb_to_date"] + other_gb, 1)
    s["headroom_gb"] = round(cap_gb - s["projected_total_gb_to_date"], 1)
    s["years_of_growth_left"] = round(s["headroom_gb"] / (day_bytes * 365.25 / 1e9), 2) if day_bytes else None
    return s


def main(env=None):
    env = env if env is not None else os.environ
    try:
        day = parse_day(env.get("IN_DAY"), "day", utcnow().date())
    except PlanError as err:
        print(f"dryplan: {err}", file=sys.stderr)
        return 2
    cfg = load_config()
    http = Http(cfg)
    out = plan_day(cfg, http, day, sample=env.get("SAMPLE", "1") == "1",
                   sample_hour=int(env.get("SAMPLE_HOUR", "21")), stations=env.get("STATIONS", "1") == "1")
    out["summary"] = summarise(out, cfg, float(env.get("CAP_GB", "2500")), float(env.get("OTHER_LANES_GB", "310")))
    out["http"] = {"requests": http.stats["requests"], "by_status": http.stats["by_status"]}
    for oid, r in sorted(out["outputs"].items()):
        smp = r.get("sample") or {}
        log("plan", oid, r["ver"], f"steps={r.get('steps_found')}/{r.get('steps_expected')}",
            f"source_mb={(r.get('source_bytes') or 0) / 1e6:.1f}", f"stored_mb={(r.get('stored_bytes') or 0) / 1e6:.1f}",
            *(f"{k}={smp[k]}" for k in ("mode", "decode_back", "max_quant_err", "exact_cells", "total_cells",
                                         "station_points", "zstd_ratio") if k in smp))
    print(canonical_json(out).decode("ascii"), flush=True)
    bad = [k for k, r in out["outputs"].items() if (r.get("sample") or {}).get("decode_back") == "fail"]
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
