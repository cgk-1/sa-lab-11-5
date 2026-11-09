"""``feed-day/1`` manifests: one immutable JSON document per day per job.

A manifest lists, for each dataset, every source (URL, byte range, sha256, ETag, Last-Modified, HTTP status, tries)
and every stored object (key, bytes, sha256, MD5). An exact-grid object also carries, per step, the GRIB packing
factors (template, R, E, D, bits) and the sha256 of its integers, of the decoded values and of the source bytes, so
each step can be proven exact without opening the object. A gridded output that uses station cells carries the
station map (station id, projected nearest cell, squared-degree nearest cell). No observation value or statistic of one is recorded.
"""

import datetime as _dt
import json
import re

from . import canonical_json, sha256_hex
from .store import PREFIXES

SCHEMA = "feed-day/1"
STATUSES = ("complete", "partial", "missing", "unchanged")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_B64MD5 = re.compile(r"^[A-Za-z0-9+/]{22}==$")
_VINTAGE = re.compile(r"^v\d{8}T\d{4}Z$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RUN_FIELDS = ("id", "attempt", "job", "lane", "git_sha", "lock_sha256", "runner_image", "started_utc", "ended_utc")
_CHECK_KEYS = {"roundtrip", "decode_back", "cells_checked", "rows", "station_points", "members", "note", "exact",
               "max_quant_err", "exact_cells", "total_cells", "empty"}
_SOURCE_KEYS = {"url", "range", "sha256", "etag", "last_modified", "valid_utc", "http", "tries", "fallback", "bytes"}
_OBJECT_KEYS = {"key", "bytes", "sha256", "md5", "content_type", "status", "covers", "exact"}


def iso(ts):
    return ts.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class DatasetEntry(dict):
    def __init__(self, id, ver, vintage):
        super().__init__(id=id, ver=ver, vintage=vintage, status="complete", sources=[], objects=[],
                         steps={"expected": 0, "got": 0, "missing": []}, checks={}, era=[], supersedes=None,
                         withdrawn=False)

    def add_source(self, url, sha256=None, http=200, tries=1, rng=None, etag=None, last_modified=None,
                   valid_utc=None, fallback=None, nbytes=None):
        s = {"url": url, "range": rng, "sha256": sha256, "etag": etag, "last_modified": last_modified,
             "valid_utc": valid_utc, "http": http, "tries": tries}
        if fallback:
            s["fallback"] = fallback
        if nbytes is not None:
            s["bytes"] = int(nbytes)
        self["sources"].append(s)
        return s

    def add_object(self, put_result, covers=None, exact=None):
        o = {k: put_result[k] for k in ("key", "bytes", "sha256", "md5", "content_type")}
        if covers:
            o["covers"] = covers
        if exact:
            o["exact"] = exact
        self["objects"].append(o)
        return o

    def add_missing(self, valid_utc, http, tries, known_gap=None):
        """``known_gap``: None for sources where a 404 is a source gap; True / False for gridded sources under the
        gap rule, where only a step on the configured known-gap list is explained."""
        m = {"valid_utc": valid_utc, "http": http, "tries": tries}
        if known_gap is not None:
            m["known_gap"] = bool(known_gap)
        self["steps"]["missing"].append(m)

    def finish_status(self):
        """complete = every expected step was got or is explained (a known gap; or, outside the gap rule, a source
        404); partial = some step is missing for another reason (or a check failed); missing = nothing was got."""
        if self["status"] == "unchanged":
            return self["status"]
        st = self["steps"]
        other = [m for m in st["missing"] if unexplained(m)]
        if st["expected"] and st["got"] == 0:
            self["status"] = "missing"
        elif other or (st["expected"] and st["got"] + len(st["missing"]) < st["expected"]):
            self["status"] = "partial"
        elif any(v in ("fail", "unavailable") for v in self["checks"].values() if isinstance(v, str)):
            self["status"] = "partial"
        else:
            self["status"] = "complete"
        return self["status"]


class DayManifest:
    def __init__(self, day, run):
        self.day = day
        self.run = dict(run)
        self.datasets = []

    def add(self, entry):
        self.datasets.append(entry)
        return entry

    def to_doc(self, ended_utc=None):
        run = dict(self.run)
        if ended_utc is not None:
            run["ended_utc"] = ended_utc
        return {"schema": SCHEMA, "day": self.day.isoformat(), "run": run, "datasets": list(self.datasets)}

    def to_bytes(self, ended_utc=None):
        doc = self.to_doc(ended_utc)
        errs = validate(doc)
        if errs:
            raise ValueError("manifest invalid: " + "; ".join(errs[:5]))
        return canonical_json(doc)


def unexplained(m):
    """A missing step that is not a known source gap. Under the gap rule (``known_gap`` present) a 404 counts too."""
    if "known_gap" in m:
        return not m["known_gap"]
    return m.get("http") != 404


def manifest_sha256(doc):
    return sha256_hex(canonical_json(doc))


def validate(doc):
    """Returns a list of problems (empty = valid)."""
    e = []
    if not isinstance(doc, dict):
        return ["not an object"]
    if doc.get("schema") != SCHEMA:
        e.append("schema")
    if not isinstance(doc.get("day"), str) or not _DAY.match(doc["day"]):
        e.append("day")
    run = doc.get("run")
    if not isinstance(run, dict):
        e.append("run")
    else:
        for f in _RUN_FIELDS:
            if f not in run:
                e.append(f"run.{f}")
        if run.get("lock_sha256") and not _HEX64.match(str(run["lock_sha256"])):
            e.append("run.lock_sha256")
    ds = doc.get("datasets")
    if not isinstance(ds, list):
        return e + ["datasets"]
    for i, d in enumerate(ds):
        p = f"datasets[{i}]"
        if not isinstance(d, dict):
            e.append(p)
            continue
        if not re.match(r"^[a-z0-9][a-z0-9-]*$", str(d.get("id", ""))):
            e.append(p + ".id")
        if not re.match(r"^v\d+(-c\d+)?$", str(d.get("ver", ""))):
            e.append(p + ".ver")
        if not _VINTAGE.match(str(d.get("vintage", ""))):
            e.append(p + ".vintage")
        if d.get("status") not in STATUSES:
            e.append(p + ".status")
        sup = d.get("supersedes")
        if sup is not None and not _VINTAGE.match(str(sup)):
            e.append(p + ".supersedes")
        if not isinstance(d.get("withdrawn"), bool):
            e.append(p + ".withdrawn")
        for j, s in enumerate(d.get("sources", [])):
            if set(s) - _SOURCE_KEYS:
                e.append(f"{p}.sources[{j}] unknown fields")
            if not str(s.get("url", "")).startswith(("https://", "s3://")):
                e.append(f"{p}.sources[{j}].url")
            if s.get("sha256") is not None and not _HEX64.match(s["sha256"]):
                e.append(f"{p}.sources[{j}].sha256")
        for j, o in enumerate(d.get("objects", [])):
            if set(o) - _OBJECT_KEYS:
                e.append(f"{p}.objects[{j}] unknown fields")
            if not str(o.get("key", "")).startswith(PREFIXES):
                e.append(f"{p}.objects[{j}].key")
            if not _HEX64.match(str(o.get("sha256", ""))):
                e.append(f"{p}.objects[{j}].sha256")
            if not _B64MD5.match(str(o.get("md5", ""))):
                e.append(f"{p}.objects[{j}].md5")
            if not isinstance(o.get("bytes"), int) or o["bytes"] < 0:
                e.append(f"{p}.objects[{j}].bytes")
        steps = d.get("steps", {})
        if not isinstance(steps, dict) or not {"expected", "got", "missing"} <= set(steps):
            e.append(p + ".steps")
        checks = d.get("checks", {})
        if set(checks) - _CHECK_KEYS:
            # Only pass/fail flags and plain counts; never a value-derived statistic.
            e.append(p + ".checks unknown fields")
    return e


def parse(data):
    doc = json.loads(data.decode("utf-8"))
    errs = validate(doc)
    if errs:
        raise ValueError("manifest invalid: " + "; ".join(errs[:5]))
    return doc


# ---------------------------------------------------------------- reading manifests back

def day_manifests(store, day):
    """All manifests stored for ``day`` (oldest key first)."""
    if not store.online:
        return []
    d = day.isoformat()
    out = []
    for o in sorted(store.list(f"man/{d[:4]}/{d}/"), key=lambda o: o["key"]):
        data, _ = store.get(o["key"])
        try:
            out.append(parse(data))
        except ValueError:
            continue
    return out


def complete_set(manifests):
    """{(dataset id, ver)} with a ``complete`` entry among ``manifests``."""
    return {(d["id"], d["ver"]) for m in manifests for d in m["datasets"]
            if d["status"] == "complete" and not d.get("withdrawn")}


def newest_entry(manifests, dataset_id, ver):
    """The newest (by vintage) non-withdrawn entry for a dataset, or None."""
    best = None
    for m in manifests:
        for d in m["datasets"]:
            if d["id"] == dataset_id and d["ver"] == ver and not d.get("withdrawn") and d["status"] != "unchanged":
                if best is None or d["vintage"] > best["vintage"]:
                    best = d
    return best
