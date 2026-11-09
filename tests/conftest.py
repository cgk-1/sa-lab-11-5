"""Shared fixtures: a fake web (HTTP transport), a fake clock, an in-memory S3 client, a fake GRIB decoder."""

import copy
import datetime as _dt
import io
import json
import urllib.parse
import zipfile

import numpy as np
import pytest

from feeder import grib, load_config
from feeder.net import Http
from feeder.run import Ctx
from feeder.selftest import FakeS3, synthetic_shp
from feeder.store import Store

DAY = _dt.date(2024, 4, 5)
STARTED = _dt.datetime(2024, 4, 20, 12, 7, tzinfo=_dt.timezone.utc)
NY, NX = 30, 40


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += max(0.0, s)


class FakeWeb:
    """Routes: exact URL -> (status, headers, body) or callable(url, parsed) -> tuple | None."""

    def __init__(self, clock):
        self.exact = {}
        self.fns = []
        self.log = []
        self.clock = clock
        self.starts = {}

    def add(self, url, body, status=200, headers=None):
        self.exact[url] = (status, headers or {}, body)

    def route(self, fn):
        self.fns.append(fn)

    def __call__(self, method, url, headers, timeout):
        p = urllib.parse.urlsplit(url)
        self.log.append({"url": url, "headers": dict(headers), "t": self.clock()})
        self.starts.setdefault(p.netloc, []).append(self.clock())
        hit = self.exact.get(url)
        if hit is None:
            for fn in self.fns:
                hit = fn(url, p)
                if hit is not None:
                    break
        if hit is None:
            return 404, {}, b"not found"
        status, hdrs, body = hit
        if callable(body):
            body = body()
        rng = headers.get("Range")
        if status == 200 and rng and body is not None:
            a, b = rng.split("=")[1].split("-")
            a = int(a)
            if a >= len(body):
                return 416, {}, b""
            b = len(body) - 1 if b == "" else min(int(b), len(body) - 1)
            return 206, {**hdrs, "Content-Range": f"bytes {a}-{b}/{len(body)}", "ETag": '"etag-%d"' % len(body)}, \
                body[a:b + 1]
        return status, {"ETag": '"etag-%d"' % (len(body or b""))} | hdrs, body

    def urls(self, host=None):
        return [x["url"] for x in self.log if host is None or urllib.parse.urlsplit(x["url"]).netloc == host]


# ---------------------------------------------------------------- fake GRIB

def fake_message(ids, values, pack=None):
    """A fake message. With ``pack`` ({R, E, D}) it carries packed integers like a real GRIB2 message and decodes to
    ((X * 2**E) + R) * 10**-D; without it, plain float64 values and no packing description."""
    head = {"ids": list(ids), "shape": list(values.shape)}
    if pack is not None:
        v = np.asarray(values, np.float64)
        s, d = grib.power(2.0, pack["E"]), grib.power(10.0, -pack["D"])
        x = np.rint((v / d - pack["R"]) / s)
        head["pack"] = {"template": 0, "R": float(pack["R"]), "E": int(pack["E"]), "D": int(pack["D"]),
                        "nbits": 16, "bitmap": False}
        payload = x.astype("<u4").tobytes()
    else:
        payload = np.asarray(values, np.float64).tobytes()
    h = json.dumps(head).encode()
    return b"FAKEGRIB" + len(h).to_bytes(4, "big") + h + payload


def grib2_message(ids, values, pack=None, surface=(103, 10)):
    """A structurally valid GRIB2 message (sections 0-7) whose section 7 carries a fake payload."""
    import struct

    payload = fake_message(ids, values, pack)
    s1 = struct.pack(">IB", 21, 1) + bytes(16)
    s3 = struct.pack(">IB", 72, 3) + bytes(1) + struct.pack(">I", values.size) + bytes(2) + struct.pack(">H", 30) + \
        bytes(58)
    s4 = struct.pack(">IB", 34, 4) + struct.pack(">H", 0) + struct.pack(">H", 0) + bytes([ids[1], ids[2]]) + \
        bytes(11) + bytes([surface[0], 0]) + struct.pack(">I", surface[1]) + bytes(6)
    s5 = struct.pack(">IB", 21, 5) + bytes(16)
    s6 = struct.pack(">IBB", 6, 6, 255)
    s7 = struct.pack(">IB", 5 + len(payload), 7) + payload
    body = s1 + s3 + s4 + s5 + s6 + s7 + b"7777"
    return b"GRIB" + bytes(2) + bytes([ids[0], 2]) + struct.pack(">Q", 16 + len(body)) + body


def _latlons():
    j, i = np.mgrid[0:NY, 0:NX]
    return 30.0 + j * 0.1, -100.0 + i * 0.1


def _locate(lat, lon):
    j = int(round((lat - 30.0) / 0.1))
    i = int(round((lon + 100.0) / 0.1))
    return j * NX + i if 0 <= j < NY and 0 <= i < NX else None


def fake_decode(message, field, expect_shape=None):
    b = bytes(message)
    if b.startswith(b"GRIB"):
        b = b[b.index(b"FAKEGRIB"):-4]
    if not b.startswith(b"FAKEGRIB"):
        raise grib.GribError("not fake grib")
    n = int.from_bytes(b[8:12], "big")
    head = json.loads(b[12:12 + n])
    pack = head.get("pack")
    if pack is not None:
        x = np.frombuffer(b[12 + n:], "<u4").reshape(head["shape"])
        vals = grib.ints_to_values(x, pack)
    else:
        vals = np.frombuffer(b[12 + n:], np.float64).reshape(head["shape"]).copy()
    if field.get("grib") is not None and tuple(head["ids"]) != tuple(field["grib"]):
        raise grib.GribError("ids")
    if expect_shape is not None and vals.shape != tuple(expect_shape):
        raise grib.GribError("shape")
    return grib.Field(vals, grib.Grid(vals.shape, _latlons, _locate, ("fake", NY, NX)), tuple(head["ids"]), pack)


def urma_field(name, hour, seed=0):
    rng = np.random.default_rng(1000 * seed + hour)
    if name == "gust":
        v = np.round(rng.uniform(2, 14, (NY, NX)), 2)
        v[5:8, 5:10] = 27.5  # ~61.5 mph: above the 58-mph box rule at 15 cells
        return v
    if name == "wind":
        return np.round(rng.uniform(1, 10, (NY, NX)), 2)
    return np.round(rng.uniform(0, 360, (NY, NX)), 1)


URMA_PACK = {"gust": {"R": 0.0, "E": 0, "D": 2}, "wind": {"R": 0.0, "E": 0, "D": 2}, "wdir": {"R": 0.0, "E": 0, "D": 1}}


def add_urma(web, day=DAY, seed=0, missing_hours=(), no_idx=False, packed=True):
    ids = {"gust": (0, 2, 22), "wind": (0, 2, 1), "wdir": (0, 2, 0)}
    names = {"gust": "GUST:10 m above ground:anl", "wind": "WIND:10 m above ground:anl",
             "wdir": "WDIR:10 m above ground:anl"}
    for h in range(24):
        if h in missing_hours:
            continue
        key = f"urma2p5.{day:%Y%m%d}/urma2p5.t{h:02d}z.2dvaranl_ndfd.grb2_wexp"
        url = f"https://noaa-urma-pds.s3.amazonaws.com/{key}"
        body, idx, off = b"", [], 0
        for n, f in enumerate(("gust", "wind", "wdir"), start=1):
            m = (grib2_message if no_idx else fake_message)(ids[f], urma_field(f, h, seed),
                                                            URMA_PACK[f] if packed else None)
            idx.append(f"{n}:{off}:d={day:%Y%m%d}{h:02d}:{names[f]}:")
            body += m
            off += len(m)
        web.add(url, body)
        if not no_idx:
            web.add(url + ".idx", ("\n".join(idx) + "\n").encode())


# ---------------------------------------------------------------- IEM fakes

CATALOG_KEY = "src/iem-stations/v2/2024/2024-04-01/v20240401T1113Z/IA_ASOS-0123456789abcdef.geojson.zst"


def seed_catalog(s3, net="IA_ASOS", n=3):
    """A stored station catalogue, as the catalogue job leaves it before the gridded lane runs."""
    from feeder import compact

    s3.objects[CATALOG_KEY.replace("IA_ASOS", net)] = (compact.zstd(catalog_geojson(net, n)), "application/zstd")

def catalog_geojson(net, n=3):
    feats = [{"type": "Feature", "id": f"{net[:2]}{k}", "properties": {"sid": f"{net[:2]}{k}"},
              "geometry": {"type": "Point", "coordinates": [-100.0 + 0.1 * (k + 1), 30.0 + 0.1 * (k + 1)]}}
             for k in range(n)]
    return json.dumps({"type": "FeatureCollection", "features": feats}).encode()


def asos_csv(stations, days, variant=0):
    rows = ["station,valid,lon,lat,elevation,sknt,gust,metar"]
    for st in stations:
        for d in days:
            for h in (0, 12):
                rows.append(f"{st},{d} {h:02d}:53,-95.1,41.2,300,{h % 9},M,{st} {d[8:]}{h:02d}53Z AUTO {variant}KT")
    return ("\n".join(rows) + "\n").encode()


def daily_csv(net, days):
    rows = ["station,day,max_wind_gust_kts"]
    for d in days:
        rows.append(f"{net[:2]}0,{d},31")
    return ("\n".join(rows) + "\n").encode()


def lsr_csv(day):
    return (f'VALID,LAT,LON,MAG,TYPETEXT,REMARK\n{day:%Y%m%d}1800,41.1,-96.2,60,TSTM WND GST,"trees down, lines down"\n'
            ).encode()


def vtec_zip(polys):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("wwa.shp", synthetic_shp(polys))
        zf.writestr("wwa.dbf", b"")
    return buf.getvalue()


def _days_in(q):
    if "sts" in q:
        a = _dt.datetime.strptime(q["sts"][0], "%Y-%m-%dT%H:%MZ").date()
        b = _dt.datetime.strptime(q["ets"][0], "%Y-%m-%dT%H:%MZ").date() - _dt.timedelta(days=1)
    elif "hour1" in q:
        a = _dt.date(int(q["year1"][0]), int(q["month1"][0]), int(q["day1"][0]))
        b = _dt.date(int(q["year2"][0]), int(q["month2"][0]), int(q["day2"][0])) - _dt.timedelta(days=1)
    else:
        a = _dt.date(int(q["year1"][0]), int(q["month1"][0]), int(q["day1"][0]))
        b = _dt.date(int(q["year2"][0]), int(q["month2"][0]), int(q["day2"][0]))
    out, d = [], a
    while d <= b:
        out.append(d.isoformat())
        d += _dt.timedelta(days=1)
    return out


def add_iem(web, variant=0):
    poly = [[(-97.1, 35.1), (-96.9, 35.1), (-96.9, 35.3), (-97.1, 35.3)]]

    def fn(url, p):
        if p.netloc != "mesonet.agron.iastate.edu":
            return None
        q = urllib.parse.parse_qs(p.query)
        if p.path.startswith("/geojson/network/"):
            net = p.path.rsplit("/", 1)[1].split(".")[0]
            return 200, {}, catalog_geojson(net)
        if p.path == "/cgi-bin/request/asos.py":
            return 200, {}, asos_csv(q["station"], _days_in(q), variant)
        if p.path == "/cgi-bin/request/asos1min.py":
            days = _days_in(q)
            rows = ["station,station_name,valid(UTC),sknt"] + [f"{s},X,{d} 00:01,3" for s in q["station"] for d in days]
            return 200, {}, ("\n".join(rows) + "\n").encode()
        if p.path == "/cgi-bin/request/daily.py":
            return 200, {}, daily_csv(q["network"][0], _days_in(q))
        if p.path == "/cgi-bin/request/rwis.py":
            rows = ["station,valid,gust"] + [f"R1,{d} 00:00,{variant}" for d in _days_in(q)]
            return 200, {}, ("\n".join(rows) + "\n").encode()
        if p.path == "/cgi-bin/request/gis/lsr.py":
            return 200, {}, lsr_csv(_dt.date.fromisoformat(_days_in(q)[0]))
        if p.path == "/cgi-bin/request/gis/watchwarn.py":
            return 200, {}, vtec_zip([poly, poly])
        return None

    web.route(fn)


def add_other(web):
    def fn(url, p):
        if p.netloc == "www.ncei.noaa.gov":
            if url.endswith("/csvfiles/"):
                return 200, {}, b'<a href="StormEvents_details-ftp_v1.0_d2024_c20240415.csv.gz">x</a>' \
                                b'<a href="StormEvents_details-ftp_v1.0_d2019_c20240415.csv.gz">x</a>'
            return 200, {}, b"\x1f\x8bfakegzip"
        if p.netloc == "www.nhc.noaa.gov":
            if url.endswith("/hurdat/"):
                return 200, {}, b'<a href="hurdat2-1851-2023-051124.txt">x</a>'
            return 200, {}, b"AL011851, UNNAMED, 14,\n"
        if p.netloc == "ftp.nhc.noaa.gov":
            if url.endswith("/btk/"):
                return 200, {}, b'<a href="bal012024.dat">x</a>'
            return 200, {}, b"AL, 01, 2024061900, , BEST\n"
        if p.netloc == "www.spc.noaa.gov":
            if url.endswith("_rpts_filtered.csv"):
                return 200, {}, b"Time,F_Scale,Location\n"
            return None
        return None

    web.route(fn)


# ---------------------------------------------------------------- context

@pytest.fixture(autouse=True)
def no_real_network(request, monkeypatch):
    """Unit tests never reach the network; only tests marked ``live`` may."""
    if request.node.get_closest_marker("live"):
        return
    import feeder.net

    def refuse(*a, **k):
        raise AssertionError("real network access in a unit test")

    monkeypatch.setattr(feeder.net, "urllib_transport", refuse)
    monkeypatch.setattr(feeder.net.urllib.request, "urlopen", refuse)
    for n in ("S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET"):
        monkeypatch.delenv(n, raising=False)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["iem"]["states"] = ["IA", "NE"]
    return c


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def web(clock):
    return FakeWeb(clock)


@pytest.fixture
def s3():
    return FakeS3()


def make_ctx(cfg, web, clock, s3, tmp_path, write=True, lane="noaa", job="job", started=STARTED, groups=None,
             run_id="100", s3_online=True, catalog=True):
    if groups is not None:
        cfg = copy.deepcopy(cfg)
        cfg["noaa"]["groups"] = [g for g in cfg["noaa"]["groups"] if g["id"] in groups]
    http = Http(cfg, transport=web, sleep=clock.sleep, clock=clock,
                wallclock=lambda: _dt.datetime(2024, 4, 20, 12, 0, tzinfo=_dt.timezone.utc))
    if catalog and lane == "noaa" and not any(k.startswith("src/iem-stations/") for k in s3.objects):
        seed_catalog(s3)
    store = Store(s3 if s3_online else None, "bucket", write=write)
    import os

    os.environ["GITHUB_RUN_ID"] = run_id
    stage = tmp_path / f"stage-{job}-{run_id}-{started:%H%M}"
    stage.mkdir(exist_ok=True)
    return Ctx(cfg, lane, job, store, http=http, decode=fake_decode, budget_min=600, sleep=clock.sleep,
               started=started, staging=str(stage))
