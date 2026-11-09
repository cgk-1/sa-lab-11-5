"""Live check of the parallel gridded lane on real data (opt-in: FEEDER_LIVE=1).

One hour of MRMS reflectivity (6 frames, ~12 MB) and one hour of RTMA-RU gusts (4 messages and their .idx) are
fetched once and kept in memory; each group is then built twice, without and with worker processes, through the
real decoder. Both runs must store the same bytes, equal to those the serial code stored before the parallel lane
(git 238dac3). The other steps of the day are not served (they are recorded as missing). No uploads."""

import datetime as _dt
import os
import re
import threading

import pytest

from feeder import grib, load_config, noaa, par
from feeder.net import Http, urllib_transport
from feeder.run import Ctx, noaa_groups
from feeder.store import Store

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("FEEDER_LIVE") != "1", reason="live test is opt-in")]

DAY = _dt.date(2024, 5, 7)
HOUR = 21
LIMIT = 200 * 1024 * 1024
GOLDEN = {
    "mrms-refl": {"refl-x-20240507.xgrid": "dd541e90ea842418622aa62ce718282b386ed22b1649cb2bd3bc60de04a11d2d"},
    "rtma-ru": {"ru-gust-x-20240507.xgrid": "36f7aac8824f31d7e42439a1ec19888abfbf4e660846f21f008e61c41ebbf705"},
}


class OneHour:
    """Real transport for the steps of one hour (memoised), 404 for every other step; listings keep that hour."""

    def __init__(self):
        ymd = f"{DAY:%Y%m%d}"
        self.hour = re.compile(rf"({ymd}-{HOUR:02d}\d{{4}}|\.t{HOUR:02d}\d\dz\.)")
        self.memo, self.bytes, self.lock = {}, 0, threading.Lock()

    def __call__(self, method, url, headers, timeout):
        listing = "?list-type" in url
        if not listing and not self.hour.search(url):
            return 404, {}, b""
        key = (url, headers.get("Range"))
        with self.lock:
            hit = self.memo.get(key)
        if hit is None:
            hit = urllib_transport(method, url, headers, timeout)
            with self.lock:
                self.memo[key] = hit
                self.bytes += len(hit[2] or b"")
                assert self.bytes < LIMIT, "live download budget exceeded"
        status, hdrs, body = hit
        if listing:
            body = re.sub(rb"<Contents>(?:(?!</Contents>).)*</Contents>",
                          lambda m: m.group(0) if self.hour.search(m.group(0).decode()) else b"", body, flags=re.S)
        return status, hdrs, body


@pytest.fixture(scope="module")
def web():
    return OneHour()


def _build(group_id, web, tmp_path, procs, monkeypatch):
    monkeypatch.setenv("FEEDER_PROCS", str(procs))
    par.shutdown()
    cfg = load_config()
    stage = tmp_path / f"{group_id}-{procs}"
    stage.mkdir()
    ctx = Ctx(cfg, "noaa", "live-par", Store(None, None, write=False), http=Http(cfg, transport=web),
              decode=grib.decode, budget_min=60, staging=str(stage))
    g = next(x for x in noaa_groups(ctx) if x["id"] == group_id)
    res = noaa.process_group(ctx, g, DAY, g["outputs"])
    ctx.close()
    assert not any(e["checks"].get("decode_back") == "fail" for e in res.entries)
    return {pr["key"].rsplit("/", 1)[1]: pr["sha256"] for pr, _ in res.written}


@pytest.mark.parametrize("group_id", ["mrms-refl", "rtma-ru"])
def test_one_real_hour_same_bytes(group_id, web, tmp_path, monkeypatch):
    serial = _build(group_id, web, tmp_path, 1, monkeypatch)
    workers = _build(group_id, web, tmp_path, max(2, os.cpu_count() or 2), monkeypatch)
    assert workers == serial == GOLDEN[group_id]
