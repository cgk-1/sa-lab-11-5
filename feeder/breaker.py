"""Run ledger and stop rules.

Each job appends events (never overwrites) under ``run/<run_id>/<job>/<seq>.json``; every event carries the job's
cumulative totals, so the run's state is the sum of the newest event of each job. Before each day a job reads the
run state and stops cleanly if any rule has tripped.

Rules:
- integrity: a decode-back / round-trip / sha mismatch fails the day at once; the run stops when more than 0.5 % of
  verified files fail;
- failures: the run stops when more than 1 % of days fail after their retries;
- size: projection = bytes so far / days done * total days + the lane's fixed tiers. Above ``cut_gb`` the next
  enabled cut is applied to days not yet written (one cut per ``size_min_days`` days); above ``stop_gb`` with no cut
  left, the run stops. A job stops when its bytes per day exceed ``bytes_per_day_factor`` x the expected rate on
  ``bytes_per_day_streak`` days in a row;
- gaps: a gridded day with more than ``gap_unexplained_max`` missing steps that are not a source 404 fails;
- observation lane: repeated 429 / 503 bursts pause the lane; more than 5 % failed requests stop it;
- prefixes: a key outside the archive prefixes (store.PrefixError) stops the run.
"""

import json

from . import canonical_json, log, utcnow
from .store import run_key


class BreakerTripped(Exception):
    def __init__(self, reason, scope="run", origin=True):
        super().__init__(reason)
        self.reason = reason
        self.scope = scope
        self.origin = origin


_ZERO = {"days_done": 0, "days_failed": 0, "bytes": 0, "files_verified": 0, "files_failed": 0, "unchanged": 0,
         "cut_level": 0, "cut_at_days": 0, "tripped": None}


class Ledger:
    def __init__(self, store, run_id, job):
        self.store = store
        self.run_id = str(run_id)
        self.job = str(job)
        self.seq = 0
        self.totals = dict(_ZERO)
        self._seen = {}

    def record(self, kind, **extra):
        self.seq += 1
        ev = {"kind": kind, "job": self.job, "seq": self.seq, "utc": utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
              "totals": dict(self.totals)}
        ev.update(extra)
        if self.store.write:
            self.store.put(run_key(self.run_id, self.job, self.seq), canonical_json(ev), "application/json")
        return ev

    def run_totals(self):
        """Sum of the newest event of every job in the run (this job's in-memory totals win for itself)."""
        latest = {}
        if self.store.online:
            for o in self.store.list(f"run/{self.run_id}/"):
                parts = o["key"].split("/")
                if len(parts) != 4:
                    continue
                job, seq = parts[2], parts[3]
                if job not in latest or seq > latest[job][0]:
                    latest[job] = (seq, o["key"])
        agg = dict(_ZERO)
        agg["tripped"] = None
        for job, (seq, key) in latest.items():
            if job == self.job:
                continue
            if self._seen.get(job, ("",))[0] != seq:
                data, _ = self.store.get(key)
                self._seen[job] = (seq, json.loads(data.decode("utf-8")))
            ev = self._seen[job][1]
            self._add(agg, ev["totals"])
        self._add(agg, self.totals)
        return agg

    @staticmethod
    def _add(agg, t):
        for k in ("days_done", "days_failed", "bytes", "files_verified", "files_failed", "unchanged"):
            agg[k] += int(t.get(k, 0))
        if int(t.get("cut_level", 0)) > agg["cut_level"]:
            agg["cut_level"] = int(t["cut_level"])
            agg["cut_at_days"] = int(t.get("cut_at_days", 0))
        if t.get("tripped") and not agg["tripped"]:
            agg["tripped"] = t["tripped"]


class Breaker:
    def __init__(self, cfg, cuts, lane, ledger, sleep=None):
        self.cfg = cfg
        self.cuts = [c for c in cuts if c.get("enabled")]
        self.lane = lane
        self.ledger = ledger
        self.sleep = sleep
        self.streak = 0
        self.size_min_days = int(cfg.get("size_min_days", 3))

    # ---- state
    def active_cuts(self):
        return self.cuts[: self.ledger.totals["cut_level"]]

    def projection_gb(self, agg):
        if agg["days_done"] <= 0:
            return None
        per_day = agg["bytes"] / agg["days_done"]
        fixed = float(self.cfg.get("fixed_gb_by_lane", {}).get(self.lane, 0.0))
        return per_day * float(self.cfg["total_days"]) / 1e9 + fixed

    def trip(self, reason, scope="run"):
        if scope in ("run", "lane"):
            self.ledger.totals["tripped"] = reason
        self.ledger.record("tripped", reason=reason, scope=scope)
        log("BREAKER", scope, reason)
        raise BreakerTripped(reason, scope)

    def check_before_day(self):
        agg = self.ledger.run_totals()
        if agg["tripped"]:
            log("BREAKER", "run already stopped:", agg["tripped"])
            raise BreakerTripped(agg["tripped"], "run", origin=False)
        if self.ledger.totals["cut_level"] < agg["cut_level"]:
            self.ledger.totals["cut_level"] = agg["cut_level"]
            self.ledger.totals["cut_at_days"] = agg["cut_at_days"]
        if agg["files_verified"] and agg["files_failed"] / agg["files_verified"] > self.cfg["integrity_fail_share"]:
            self.trip(f"integrity: {agg['files_failed']} of {agg['files_verified']} verified files failed")
        if agg["days_done"] and agg["days_failed"] / agg["days_done"] > self.cfg["day_fail_share"]:
            self.trip(f"failures: {agg['days_failed']} of {agg['days_done']} days failed")
        proj = self.projection_gb(agg) if self.lane != "pilot" else None
        if proj is not None and agg["days_done"] >= self.size_min_days:
            if proj > self.cfg["cut_gb"]:
                level = self.ledger.totals["cut_level"]
                since = agg["days_done"] - self.ledger.totals["cut_at_days"]
                if level < len(self.cuts) and (level == 0 or since >= self.size_min_days):
                    self.ledger.totals["cut_level"] = level + 1
                    self.ledger.totals["cut_at_days"] = agg["days_done"]
                    self.ledger.record("cut", cut=self.cuts[level]["id"], projection_gb=round(proj, 1))
                    log("cut", self.cuts[level]["id"], f"projection_gb={proj:.1f}")
                elif level >= len(self.cuts) and proj > self.cfg["stop_gb"]:
                    self.trip(f"size: projection {proj:.1f} GB above {self.cfg['stop_gb']} GB after every cut")
        return agg

    def after_day(self, day, nbytes, failed, verified=0, verify_failed=0, unchanged=0):
        t = self.ledger.totals
        t["days_done"] += 1
        t["days_failed"] += 1 if failed else 0
        t["bytes"] += int(nbytes)
        t["files_verified"] += int(verified)
        t["files_failed"] += int(verify_failed)
        t["unchanged"] += int(unchanged)
        self.ledger.record("day", day=day.isoformat(), bytes=int(nbytes), failed=bool(failed))
        exp = (self.cfg.get("expected_bytes_per_day") or {}).get(self.lane)
        if exp:
            if nbytes > self.cfg["bytes_per_day_factor"] * exp:
                self.streak += 1
            else:
                self.streak = 0
            if self.streak >= self.cfg["bytes_per_day_streak"]:
                self.trip(f"size: bytes per day above {self.cfg['bytes_per_day_factor']}x expected on "
                          f"{self.streak} days in a row", scope="job")

    def gap_failed(self, missing):
        unexplained = [m for m in missing if m.get("http") != 404]
        return len(unexplained) > int(self.cfg["gap_unexplained_max"])

    def observe_http(self, http):
        """Observation-lane politeness: pause after repeated bursts; stop when too many requests fail."""
        if http.recent_bursts(self.cfg["iem_burst_window_s"]) >= self.cfg["iem_burst_limit"]:
            log("pause", f"seconds={self.cfg['iem_pause_s']}", "reason=repeated 429/503")
            http.stats["bursts"].clear()
            if self.sleep:
                self.sleep(self.cfg["iem_pause_s"])
        req, fail = http.stats["requests"], http.stats["failed"]
        if req >= self.cfg["iem_min_requests"] and fail / req > self.cfg["iem_fail_share"]:
            self.trip(f"observation lane: {fail} of {req} requests failed", scope="lane")
