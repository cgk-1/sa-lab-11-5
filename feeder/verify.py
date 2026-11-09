"""Re-verification: HEAD every object listed in the manifests (length must match), GET + sha256 on a seeded sample,
a 206 range read of every PMTiles and exact-grid header (header failures of both count under ``pmtiles_fail``), and a list of orphans (stored objects named in no manifest). Never
deletes anything. HEAD and ranged GET work the same on objects in the Infrequent Access class (a GET of one is billed
as a retrieval; the sample keeps those few)."""

import concurrent.futures as cf
import hashlib
import random

from . import log
from .manifest import day_manifests


VERIFY_THREADS = 8   # object-store requests in flight (HEAD and range reads)
VERIFY_GETS = 3      # whole-object GETs in flight (each held in memory while hashed)


def _head_ok(store, o):
    h = store.head(o["key"])
    return h is not None and h["bytes"] == o["bytes"]


def _get_ok(store, o):
    data, _ = store.get(o["key"])
    return hashlib.sha256(data).hexdigest() == o["sha256"]


def _header_ok(store, key):
    try:
        if key.endswith(".xgrid"):
            head, status = store.get(key, (0, 23))
            return status == 206 and head[:8] in (b"XGRID001", b"XGRID002")  # format 1 and format 2 objects
        head, status = store.get(key, (0, 126))
        return status == 206 and head[:7] == b"PMTiles" and head[7] == 3
    except Exception:
        return False


def verify_objects(store, objects, share=0.02, min_n=3, seed=0, check_all=False):
    """The checks run concurrently; the report (counts and the sorted failed keys) is the same as one by one."""
    rep = {"objects": len(objects), "head_fail": 0, "get_checked": 0, "get_fail": 0, "pmtiles_checked": 0,
           "pmtiles_fail": 0, "failed_keys": []}
    if not store.online:
        rep["note"] = "offline"
        return rep
    n = len(objects) if check_all else min(len(objects), max(min_n, int(round(len(objects) * share))))
    sample = random.Random(seed).sample(list(objects), n) if n < len(objects) else list(objects)
    headers = [k for k, o in enumerate(objects) if o["key"].endswith((".xgrid", ".pmtiles"))]
    with cf.ThreadPoolExecutor(max_workers=VERIFY_THREADS) as pool, \
            cf.ThreadPoolExecutor(max_workers=VERIFY_GETS) as gets:
        heads = list(pool.map(lambda o: _head_ok(store, o), objects))
        got = list(gets.map(lambda o: _get_ok(store, o), sample))
        hdr = dict(zip(headers, pool.map(lambda k: _header_ok(store, objects[k]["key"]), headers)))
    for o, ok in zip(objects, heads):
        if not ok:
            rep["head_fail"] += 1
            rep["failed_keys"].append(o["key"])
    for o, ok in zip(sample, got):
        rep["get_checked"] += 1
        if not ok:
            rep["get_fail"] += 1
            rep["failed_keys"].append(o["key"])
    for k, o in enumerate(objects):
        if o["key"].endswith(".xgrid"):
            rep["xgrid_checked"] = rep.get("xgrid_checked", 0) + 1
        if o["key"].endswith(".pmtiles"):
            rep["pmtiles_checked"] += 1
        if k in hdr and not hdr[k]:
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
