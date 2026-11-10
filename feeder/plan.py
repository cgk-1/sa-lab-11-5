"""Validates the dispatch inputs and prints the job matrices as GITHUB_OUTPUT lines.

Inputs (environment only, never interpolated into shell text): IN_LANE, IN_START, IN_END, IN_DAYS, IN_DPJ, IN_PAR.
Rules: days match ^\\d{4}-\\d{2}-\\d{2}$ and fall in [archive start, yesterday UTC]; the observation lane is always
month chunks with one job at a time; the mesonet lane (madis) is month chunks from its first day (MADIS_START);
pilot is one day per job, 3 in parallel, at most 10 days; every matrix holds at most 256 entries; unused lanes get [].
Standard library only (runs before dependencies are installed).
"""

import datetime as _dt
import json
import math
import os
import re
import sys

DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
LANES = ("pilot", "vtec", "noaa", "iem", "madis", "verify")
OUTPUTS = ("noaa", "iem", "verify", "madis")
MAX_MATRIX = 256
MAX_PARALLEL = 30
ARCHIVE_START = _dt.date(2021, 10, 1)
MADIS_START = _dt.date(2023, 3, 1)  # = config madis.start (a test keeps them equal; this module reads no config)


class PlanError(Exception):
    pass


def today_utc():
    return _dt.datetime.now(_dt.timezone.utc).date()


def parse_day(text, what, today):
    text = (text or "").strip()
    if not DAY_RE.match(text):
        raise PlanError(f"{what}: expected YYYY-MM-DD")
    try:
        d = _dt.date.fromisoformat(text)
    except ValueError as err:
        raise PlanError(f"{what}: not a calendar day") from err
    if d < ARCHIVE_START or d > today - _dt.timedelta(days=1):
        raise PlanError(f"{what}: outside {ARCHIVE_START} .. yesterday (UTC)")
    return d


def chunk(a, b):
    return f"{a.isoformat()}..{b.isoformat()}"


def parse_chunk(text):
    a, b = text.split("..")
    return _dt.date.fromisoformat(a), _dt.date.fromisoformat(b)


def day_chunks(start, end, per):
    out, d = [], start
    while d <= end:
        e = min(end, d + _dt.timedelta(days=per - 1))
        out.append(chunk(d, e))
        d = e + _dt.timedelta(days=1)
    return out


def month_chunks(start, end):
    out, d = [], start
    while d <= end:
        nxt = (d.replace(day=1) + _dt.timedelta(days=32)).replace(day=1)
        e = min(end, nxt - _dt.timedelta(days=1))
        out.append(chunk(d, e))
        d = e + _dt.timedelta(days=1)
    return out


def plan(env, today=None):
    today = today or today_utc()
    lane = (env.get("IN_LANE") or "").strip()
    if lane not in LANES:
        raise PlanError("lane: not one of " + ", ".join(LANES))
    out = {k: [] for k in OUTPUTS}
    out["parallel"] = 1
    if lane == "pilot":
        raw = [x.strip() for x in (env.get("IN_DAYS") or "").split(",") if x.strip()]
        if not 1 <= len(raw) <= 10:
            raise PlanError("days: 1 to 10 comma-separated days")
        days = sorted({parse_day(x, "days", today) for x in raw})
        out["noaa"] = [chunk(d, d) for d in days]
        out["iem"] = [chunk(d, d) for d in days]
        out["parallel"] = 3
        return out
    start = parse_day(env.get("IN_START"), "start", today)
    end = parse_day(env.get("IN_END"), "end", today)
    if start > end:
        raise PlanError("start after end")
    if lane == "noaa":
        dpj = (env.get("IN_DPJ") or "7").strip()
        if dpj not in ("1", "7"):
            raise PlanError("days_per_job: 1 or 7")
        try:
            par = int(str(env.get("IN_PAR") or "14").strip())
        except ValueError as err:
            raise PlanError("max_parallel: an integer") from err
        if not 1 <= par <= MAX_PARALLEL:
            raise PlanError(f"max_parallel: 1 .. {MAX_PARALLEL}")
        out["noaa"] = day_chunks(start, end, int(dpj))
        out["parallel"] = par
    elif lane in ("iem", "vtec"):
        out["iem"] = month_chunks(start, end)
    elif lane == "madis":
        if start < MADIS_START:
            raise PlanError(f"start: the mesonet lane begins {MADIS_START}")
        out["madis"] = month_chunks(start, end)
    elif lane == "verify":
        ndays = (end - start).days + 1
        per = max(7, math.ceil(ndays / MAX_MATRIX))
        out["verify"] = day_chunks(start, end, per)
    for k in OUTPUTS:
        if len(out[k]) > MAX_MATRIX:
            raise PlanError(f"{k}: {len(out[k])} jobs > {MAX_MATRIX}; split the range into smaller dispatches")
    return out


def main():
    try:
        out = plan(os.environ)
    except PlanError as err:
        print(f"plan: {err}", file=sys.stderr)
        return 1
    for k in OUTPUTS:
        print(f"{k}={json.dumps(out[k], separators=(',', ':'))}")
    print(f"parallel={int(out['parallel'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
