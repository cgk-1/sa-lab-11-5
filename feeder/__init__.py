"""Copies public weather datasets into compact, checksummed objects in S3-compatible storage."""

import base64
import datetime as _dt
import hashlib
import json
import os
import sys

__version__ = "2.0.0"

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "datasets.json")


def load_config(path=None):
    with open(path or _CONFIG_PATH, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def md5_b64(data):
    return base64.b64encode(hashlib.md5(data).digest()).decode("ascii")


def md5_hex(data):
    return hashlib.md5(data).hexdigest()


def utcnow():
    return _dt.datetime.now(_dt.timezone.utc)


def parse_day(text):
    return _dt.date.fromisoformat(text)


def day_range(start, end):
    d = start
    while d <= end:
        yield d
        d += _dt.timedelta(days=1)


def canonical_json(obj):
    """Deterministic JSON bytes (sorted keys, no whitespace), so a document's sha256 is stable."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def log(*parts):
    """Log lines are public: callers pass only keys, sizes, hashes, row counts and HTTP statuses."""
    print(" ".join(str(p) for p in parts), file=sys.stdout, flush=True)
