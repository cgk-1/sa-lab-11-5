"""Re-verification: HEAD every object listed in the manifests (length must match), GET + sha256 on a seeded sample,
a 206 range read of every PMTiles and exact-grid header (header failures of both count under ``pmtiles_fail``), and a list of orphans (stored objects named in no manifest). Never
deletes anything."""

import hashlib
import random

from . import log
from .manifest import day_manifests


def verify_objects(store, objects, share=0.02, min_n=3, seed=0, check_all=False):
    rep = {"objects": len(objects), "head_fail": 0, "get_checked": 0, "get_fail": 0, "pmtiles_checked": 0,
           "pmtiles_fail": 0, "failed_keys": []}
    if not store.online:
        rep["note"] = "offline"
        return rep
    for o in objects:
        h = store.head(o["key"])
        if h is None or h["bytes"] != o["bytes"]:
            rep["head_fail"] += 1
            rep["failed_keys"].append(o["key"])
    n = len(objects) if check_all else min(len(objects), max(min_n, int(round(len(objects) * share))))
    sample = random.Random(seed).sample(list(objects), n) if n < len(objects) else list(objects)
    for o in sample:
        data, _ = store.get(o["key"])
        rep["get_checked"] += 1
        if hashlib.sha256(data).hexdigest() != o["sha256"]:
            rep["get_fail"] += 1
            rep["failed_keys"].append(o["key"])
    for o in objects:
        if o["key"].endswith(".xgrid"):
            rep["xgrid_checked"] = rep.get("xgrid_checked", 0) + 1
            try:
                head, status = store.get(o["key"], (0, 23))
                ok = status == 206 and head[:8] == b"XGRID001"
            except Exception:
                ok = False
            if not ok:
                rep["pmtiles_fail"] += 1
                rep["failed_keys"].append(o["key"])
        if o["key"].endswith(".pmtiles"):
            rep["pmtiles_checked"] += 1
            try:
                head, status = store.get(o["key"], (0, 126))
                ok = status == 206 and head[:7] == b"PMTiles" and head[7] == 3
            except Exception:
                ok = False
            if not ok:
                rep["pmtiles_fail"] += 1
                rep["failed_keys"].append(o["key"])
    rep["failed_keys"] = sorted(set(rep["failed_keys"]))
    return rep


def failures(rep):
    return rep["head_fail"] + rep["get_fail"] + rep["pmtiles_fail"]


def checked(rep):
    return rep["objects"] + rep["get_checked"] + rep["pmtiles_checked"]


def orphans(store, day, manifests, datasets):
    """Objects for ``day`` under src/ and tiles/ that no manifest lists. ``datasets``: [(id, ver, kind)]."""
    listed = {o["key"] for m in manifests for d in m["datasets"] for o in d["objects"]}
    d = day.isoformat()
    found = set()
    for ds_id, ver, kind in datasets:
        if kind == "tile":
            for o in store.list(f"tiles/{ds_id}/{ver}/{d[:4]}/"):
                if o["key"].split("/")[-1].startswith(d):
                    found.add(o["key"])
        else:
            for o in store.list(f"src/{ds_id}/{ver}/{d[:4]}/{d}/"):
                found.add(o["key"])
    return sorted(found - listed)


def verify_day(store, day, datasets, share, min_n, seed, check_all=False):
    mans = day_manifests(store, day)
    objs = {o["key"]: o for m in mans for ds in m["datasets"] for o in ds["objects"]}
    rep = verify_objects(store, list(objs.values()), share, min_n, seed, check_all)
    rep["day"] = day.isoformat()
    rep["manifests"] = len(mans)
    rep["orphans"] = orphans(store, day, mans, datasets) if store.online else []
    log("verify", day.isoformat(), f"manifests={len(mans)}", f"objects={rep['objects']}",
        f"fail={failures(rep)}", f"orphans={len(rep['orphans'])}")
    return rep
