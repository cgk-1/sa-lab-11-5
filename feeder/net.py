"""HTTP GET with retries, back-off, byte ranges, per-host concurrency caps and request spacing.

Politeness rules (from config/datasets.json ``hosts``):
- ``max_inflight``: requests in flight to one host from this process (1 for the observation archive);
- ``min_interval_s``: minimum spacing between request starts to one host (and per path, ``path_min_interval_s``);
- 429 / 503: exponential back-off from ``backoff_start_s`` doubling to ``backoff_cap_s``;
- ``quiet_utc``: a daily window in which no request is sent to that host.
The User-Agent is neutral and carries a contact address from the CONTACT_EMAIL repository variable.
"""

import datetime as _dt
import hashlib
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from . import log

RETRY_STATUSES = {429, 500, 502, 503, 504}
BURST_STATUSES = {429, 503}


class HttpError(Exception):
    def __init__(self, status, url, tries):
        super().__init__(f"HTTP {status} after {tries} tries: {short_url(url)}")
        self.status = status
        self.url = url
        self.tries = tries


class Response:
    def __init__(self, url, status, headers, body, tries, rng=None):
        self.url = url
        self.status = status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self.body = body
        self.tries = tries
        self.range = rng

    @property
    def ok(self):
        return self.status in (200, 206)

    @property
    def sha256(self):
        return hashlib.sha256(self.body).hexdigest() if self.body is not None else None

    @property
    def etag(self):
        v = self.headers.get("etag")
        return v.strip('"') if v else None

    @property
    def last_modified(self):
        return self.headers.get("last-modified")

    def source_record(self, valid_utc=None, fallback=None):
        """Keyword arguments for manifest.DatasetEntry.add_source."""
        return {"url": self.url, "rng": self.range, "sha256": self.sha256 if self.ok else None,
                "etag": self.etag, "last_modified": self.last_modified, "valid_utc": valid_utc,
                "http": self.status, "tries": self.tries, "fallback": fallback,
                "nbytes": len(self.body) if (self.ok and self.body is not None) else None}


def short_url(url):
    p = urllib.parse.urlsplit(url)
    return f"{p.netloc}{p.path}"


def user_agent(product):
    contact = os.environ.get("CONTACT_EMAIL", "").strip() or "contact-unset"
    return f"{product} (contact: {contact})"


def urllib_transport(method, url, headers, timeout):
    """Default transport: stdlib urllib (honours HTTPS_PROXY and the system CA store)."""
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, dict(r.headers.items()), r.read()
    except urllib.error.HTTPError as err:
        try:
            body = err.read()
        except Exception:
            body = b""
        return err.code, dict(err.headers.items()) if err.headers else {}, body


class Http:
    def __init__(self, config, transport=None, sleep=time.sleep, clock=time.monotonic, wallclock=None,
                 timeout=300):
        self.hosts = config.get("hosts", {})
        self.path_interval = config.get("path_min_interval_s", {})
        self.ua = user_agent(config.get("user_agent", "public-data-mirror/1.0"))
        self.transport = transport or urllib_transport
        self.sleep = sleep
        self.clock = clock
        self.wallclock = wallclock or (lambda: _dt.datetime.now(_dt.timezone.utc))
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sems = {}
        self._last = {}
        self.stats = {"requests": 0, "failed": 0, "bursts": [], "by_status": {}}

    # ---- policy
    def policy(self, host):
        return {**self.hosts.get("*", {}), **self.hosts.get(host, {})}

    def _sem(self, host):
        with self._lock:
            if host not in self._sems:
                self._sems[host] = threading.BoundedSemaphore(int(self.policy(host).get("max_inflight", 1)))
            return self._sems[host]

    def _space(self, slot, interval):
        if interval <= 0:
            return
        while True:
            with self._lock:
                now = self.clock()
                last = self._last.get(slot)
                wait = 0 if last is None else last + interval - now
                if wait <= 0:
                    self._last[slot] = now
                    return
            self.sleep(wait)

    def _quiet_wait(self, host):
        q = self.policy(host).get("quiet_utc")
        if not q:
            return
        now = self.wallclock()
        h0, m0 = (int(x) for x in q[0].split(":"))
        h1, m1 = (int(x) for x in q[1].split(":"))
        start = now.replace(hour=h0, minute=m0, second=0, microsecond=0)
        end = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
        if start <= now < end:
            wait = (end - now).total_seconds()
            log("quiet-window", host, f"wait_s={int(wait)}")
            self.sleep(wait)

    # ---- request
    def get(self, url, rng=None, allow=(404,), headers=None):
        """GET ``url``. ``rng`` = (start, end) inclusive, or (start, None) for an open range.
        Returns a Response. Statuses in ``allow`` are returned without retry; other failures raise HttpError."""
        parts = urllib.parse.urlsplit(url)
        if parts.scheme != "https":
            raise ValueError("https only")
        host = parts.netloc
        pol = self.policy(host)
        retries = int(pol.get("retries", 3))
        hdrs = {"User-Agent": self.ua, "Accept-Encoding": "identity"}
        if headers:
            hdrs.update(headers)
        rng_text = None
        if rng is not None:
            rng_text = f"bytes={int(rng[0])}-" + ("" if rng[1] is None else str(int(rng[1])))
            hdrs["Range"] = rng_text
        backoff = float(pol.get("backoff_start_s", 2))
        cap = float(pol.get("backoff_cap_s", 60))
        tries = 0
        last_status = None
        while True:
            tries += 1
            self._quiet_wait(host)
            sem = self._sem(host)
            with sem:
                self._space(("host", host), float(pol.get("min_interval_s", 0)))
                pi = self.path_interval.get(parts.path)
                if pi:
                    self._space(("path", host, parts.path), float(pi))
                try:
                    status, rh, body = self.transport("GET", url, hdrs, self.timeout)
                except Exception as err:  # network error
                    status, rh, body = None, {}, None
                    log("net-error", short_url(url), type(err).__name__)
            self.stats["requests"] += 1
            self.stats["by_status"][str(status)] = self.stats["by_status"].get(str(status), 0) + 1
            last_status = status
            if status in (200, 206):
                if rng is not None and status == 200:
                    # Server ignored the range: slice it ourselves so the bytes match the request.
                    end = None if rng[1] is None else int(rng[1]) + 1
                    body = body[int(rng[0]):end]
                return Response(url, status, rh, body, tries, rng_text)
            if status in allow:
                return Response(url, status, rh, None, tries, rng_text)
            if status in BURST_STATUSES:
                self.stats["bursts"].append(self.clock())
            retryable = status is None or status in RETRY_STATUSES
            if not retryable or tries >= retries:
                self.stats["failed"] += 1
                raise HttpError(last_status, url, tries)
            wait = min(backoff, cap)
            log("retry", short_url(url), f"http={status}", f"wait_s={int(wait)}")
            self.sleep(wait)
            backoff = min(backoff * 2, cap)

    def fetch(self, url, rng=None, allow=(404,), headers=None):
        """Like get(), but a request that still fails after its retries comes back as a Response with no body
        and the last status (0 = network error) instead of raising, so the caller can list it as missing."""
        try:
            return self.get(url, rng=rng, allow=allow, headers=headers)
        except HttpError as err:
            rng_text = None if rng is None else f"bytes={int(rng[0])}-" + ("" if rng[1] is None else str(int(rng[1])))
            return Response(url, err.status or 0, {}, None, err.tries, rng_text)

    def recent_bursts(self, window_s):
        now = self.clock()
        return sum(1 for t in self.stats["bursts"] if now - t <= window_s)
