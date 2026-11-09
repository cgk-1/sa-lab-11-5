"""Object storage writes that never overwrite.

Every PUT carries ``If-None-Match: *`` and ``Content-MD5``. A 412 (or 409 from a bucket lock) on an existing key is
resolved by a HEAD: an equal ETag (MD5) counts the object as already done; a different one stops the run.
Keys outside the archive prefixes are refused. Credentials, bucket and endpoint come only from the environment and
are never printed.
"""

import datetime as _dt
import hashlib
import os
import re

from . import log, md5_b64, md5_hex

PREFIXES = ("src/", "tiles/", "man/", "vrf/", "idx/", "run/")
_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~/-]*$")
_VINTAGE_RE = re.compile(r"^v\d{8}T\d{4}Z$")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_VER_RE = re.compile(r"^v\d+(-c\d+)?$")
MAX_SINGLE_PUT = 4_900_000_000



def endpoint_base(url):
    """The S3 endpoint as scheme://host only. A pasted endpoint that carries the bucket as a path
    (https://<account>.r2.cloudflarestorage.com/<bucket>) would break request signing, so the path is dropped."""
    from urllib.parse import urlsplit
    u = urlsplit(url.strip())
    if not u.scheme or not u.netloc:
        raise StoreError("S3_ENDPOINT is not a URL of the form https://host")
    if u.path.strip("/"):
        print("note: S3_ENDPOINT carried a path; using scheme://host only")
    return f"{u.scheme}://{u.netloc}"


def settings_check(vals):
    """Yes/no facts about the four storage settings, safe for a public log (no values, no lengths)."""
    import re
    from urllib.parse import urlsplit
    host = urlsplit(vals.get("S3_ENDPOINT", "").strip()).netloc
    facts = {
        "endpoint is an R2 account host": bool(re.fullmatch(r"[0-9a-f]{32}\.r2\.cloudflarestorage\.com", host)),
        "endpoint had a path": bool(urlsplit(vals.get("S3_ENDPOINT", "").strip()).path.strip("/")),
        "key id is 32 hex": bool(re.fullmatch(r"[0-9a-f]{32}", vals.get("S3_KEY_ID", ""))),
        "secret is 64 hex": bool(re.fullmatch(r"[0-9a-f]{64}", vals.get("S3_SECRET", ""))),
    }
    return "; ".join(f"{k}: {'yes' if v else 'no'}" for k, v in facts.items())

class StoreError(Exception):
    pass


class PrefixError(StoreError):
    """A key outside the archive prefixes: fatal for the run."""


class OverwriteRefused(StoreError):
    """An existing key with different content: fatal for the run (never overwrite)."""


class IntegrityError(StoreError):
    """The stored object does not match what was sent."""


# ---------------------------------------------------------------- keys

def vintage_of(when):
    """Vintage folder name for a pull that started at ``when`` (UTC)."""
    when = when.astimezone(_dt.timezone.utc)
    return "v" + when.strftime("%Y%m%dT%H%MZ")


def _check_name(name, what):
    if not _NAME_RE.match(name):
        raise PrefixError(f"bad {what}: {name!r}")


def _check_ver(ver):
    if not _VER_RE.match(ver):
        raise PrefixError(f"bad version: {ver!r}")


def src_key(dataset, ver, day, vintage, filename):
    _check_name(dataset, "dataset")
    _check_ver(ver)
    if not _VINTAGE_RE.match(vintage):
        raise PrefixError(f"bad vintage: {vintage!r}")
    if "/" in filename or not _KEY_RE.match(filename):
        raise PrefixError(f"bad file name: {filename!r}")
    d = day.isoformat()
    return f"src/{dataset}/{ver}/{d[:4]}/{d}/{vintage}/{filename}"


def tile_key(layer, ver, day, revision=0):
    _check_name(layer, "layer")
    _check_ver(ver)
    d = day.isoformat()
    suffix = "" if revision == 0 else f"~r{int(revision)}"
    return f"tiles/{layer}/{ver}/{d[:4]}/{d}{suffix}.pmtiles"


def _safe(part):
    s = re.sub(r"[^A-Za-z0-9._-]", "-", str(part))
    return s.strip("-") or "x"


def man_key(day, run_id, attempt, job):
    d = day.isoformat()
    return f"man/{d[:4]}/{d}/{_safe(run_id)}-{_safe(attempt)}-{_safe(job)}.json"


def vrf_key(day, run_id, attempt, job):
    return f"vrf/{day.isoformat()}/{_safe(run_id)}-{_safe(attempt)}-{_safe(job)}.json"


def idx_key(ver, build_day, run_id, attempt):
    _check_ver(ver)
    return f"idx/{ver}/{build_day.isoformat()}/{_safe(run_id)}-{_safe(attempt)}.json"


def run_key(run_id, job, seq):
    return f"run/{_safe(run_id)}/{_safe(job)}/{int(seq):06d}.json"


def check_key(key):
    if not isinstance(key, str) or not key.startswith(PREFIXES) or not _KEY_RE.match(key) or ".." in key \
            or "//" in key or len(key) > 900:
        raise PrefixError(f"key outside the archive prefixes: {key!r}")
    return key


# ---------------------------------------------------------------- store

def _status_of(err):
    resp = getattr(err, "response", None) or {}
    meta = resp.get("ResponseMetadata", {}) if isinstance(resp, dict) else {}
    code = meta.get("HTTPStatusCode")
    if code is None:
        try:
            code = int(resp.get("Error", {}).get("Code"))
        except (TypeError, ValueError, AttributeError):
            code = None
    return code


class PutResult(dict):
    """{key, bytes, sha256, md5, content_type, status}; status is written | exists-identical | dry-run."""


class Store:
    def __init__(self, client=None, bucket=None, write=False):
        self._client = client
        self._bucket = bucket
        self.write = bool(write)
        self.puts = 0
        self.bytes_written = 0

    @classmethod
    def from_env(cls, write):
        """Builds the client from S3_ENDPOINT / S3_BUCKET / S3_KEY_ID / S3_SECRET. Without them only a dry run
        (no reads, no writes) is possible."""
        names = ("S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET")
        vals = {n: (os.environ.get(n) or "").strip() for n in names}
        have = all(vals.values())
        if not have:
            if write:
                raise StoreError("storage credentials are not set; a writing run needs them")
            return cls(None, None, write=False)
        print("storage settings check: " + settings_check(vals))
        import boto3
        from botocore.config import Config

        cfg = Config(
            region_name="auto",
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            s3={"addressing_style": "path"},
        )
        client = boto3.client(
            "s3",
            endpoint_url=endpoint_base(vals["S3_ENDPOINT"]),
            aws_access_key_id=vals["S3_KEY_ID"],
            aws_secret_access_key=vals["S3_SECRET"],
            config=cfg,
        )
        return cls(client, vals["S3_BUCKET"], write=write)

    @property
    def online(self):
        return self._client is not None

    # ---- reads
    def head(self, key):
        check_key(key)
        if not self.online:
            return None
        try:
            r = self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception as err:  # botocore ClientError
            if _status_of(err) in (404, 403) or "NoSuchKey" in str(err) or "Not Found" in str(err):
                return None
            raise
        return {"bytes": int(r.get("ContentLength", 0)), "etag": str(r.get("ETag", "")).strip('"'),
                "content_type": r.get("ContentType")}

    def get(self, key, byte_range=None):
        check_key(key)
        if not self.online:
            raise StoreError("no storage client (dry run without credentials)")
        kw = {"Bucket": self._bucket, "Key": key}
        if byte_range is not None:
            kw["Range"] = f"bytes={int(byte_range[0])}-{int(byte_range[1])}"
        r = self._client.get_object(**kw)
        return r["Body"].read(), int(r.get("ResponseMetadata", {}).get("HTTPStatusCode", 200))

    def list(self, prefix):
        check_key(prefix if prefix.endswith("/") else prefix + "/")
        if not self.online:
            return
        token = None
        while True:
            kw = {"Bucket": self._bucket, "Prefix": prefix, "MaxKeys": 1000}
            if token:
                kw["ContinuationToken"] = token
            r = self._client.list_objects_v2(**kw)
            for o in r.get("Contents", []) or []:
                yield {"key": o["Key"], "bytes": int(o.get("Size", 0)), "etag": str(o.get("ETag", "")).strip('"')}
            if not r.get("IsTruncated"):
                break
            token = r.get("NextContinuationToken")

    # ---- writes
    def put(self, key, data, content_type="application/octet-stream"):
        """Writes ``data`` (bytes) under a new key. Never overwrites."""
        check_key(key)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes")
        data = bytes(data)
        if len(data) > MAX_SINGLE_PUT:
            raise StoreError(f"object too large for a single PUT: {key}")
        res = PutResult(key=key, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), md5=md5_b64(data),
                        content_type=content_type, status="dry-run")
        if not self.write:
            return res
        if not self.online:
            raise StoreError("no storage client")
        try:
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentMD5=res["md5"],
                                    ContentType=content_type, IfNoneMatch="*")
        except Exception as err:
            code = _status_of(err)
            if code not in (409, 412):
                raise
            existing = self.head(key)
            if existing is None:
                raise StoreError(f"conditional PUT refused ({code}) but the key is absent: {key}")
            if existing["etag"].lower() == md5_hex(data) and existing["bytes"] == len(data):
                res["status"] = "exists-identical"
                log("exists", key, len(data), res["sha256"])
                return res
            raise OverwriteRefused(f"existing object differs, refusing to overwrite: {key}")
        got = self.head(key)
        if got is None or got["bytes"] != len(data):
            raise IntegrityError(f"length after PUT does not match: {key}")
        self.puts += 1
        self.bytes_written += len(data)
        res["status"] = "written"
        log("put", key, len(data), res["sha256"])
        return res
