"""Offline self-test on synthetic fixtures it generates (no real data, no network): ``python -m feeder.selftest``.

Also provides FakeS3, an in-memory S3 client with the conditional-PUT behaviour the store relies on (used by the
unit tests).
"""

import base64
import datetime as _dt
import hashlib
import io
import os
import struct
import sys
import tempfile
import zipfile

import numpy as np


# ---------------------------------------------------------------- in-memory S3

def _client_error(status, code, op):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}, "ResponseMetadata": {"HTTPStatusCode": status}}, op)


class _Body:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


class FakeS3:
    """put_object (IfNoneMatch='*' -> 412 on an existing key; ContentMD5 checked; StorageClass kept and reported by
    head_object for non-STANDARD objects), head_object, get_object (Range -> 206), list_objects_v2 (paged).
    ``locked`` prefixes refuse overwrite with 409 like a bucket lock."""

    def __init__(self, page_size=1000, locked=()):
        self.objects = {}
        self.classes = {}
        self.calls = []
        self.page_size = page_size
        self.locked = tuple(locked)

    def put_object(self, Bucket, Key, Body, ContentMD5=None, ContentType=None, IfNoneMatch=None,
                   StorageClass=None):
        self.calls.append(("put", Key))
        if StorageClass is not None and StorageClass not in ("STANDARD", "STANDARD_IA"):
            raise _client_error(400, "InvalidStorageClass", "PutObject")
        data = bytes(Body)
        if ContentMD5 is not None and base64.b64encode(hashlib.md5(data).digest()).decode() != ContentMD5:
            raise _client_error(400, "BadDigest", "PutObject")
        if Key in self.objects:
            if IfNoneMatch == "*":
                raise _client_error(412, "PreconditionFailed", "PutObject")
            if Key.startswith(self.locked):
                raise _client_error(409, "ObjectLockedByBucketPolicy", "PutObject")
        self.objects[Key] = (data, ContentType)
        self.classes[Key] = StorageClass or "STANDARD"
        return {"ETag": '"%s"' % hashlib.md5(data).hexdigest()}

    def head_object(self, Bucket, Key):
        self.calls.append(("head", Key))
        if Key not in self.objects:
            raise _client_error(404, "404", "HeadObject")
        data, ctype = self.objects[Key]
        out = {"ContentLength": len(data), "ETag": '"%s"' % hashlib.md5(data).hexdigest(), "ContentType": ctype}
        if self.classes.get(Key, "STANDARD") != "STANDARD":  # like S3: the header only for non-STANDARD objects
            out["StorageClass"] = self.classes[Key]
        return out

    def get_object(self, Bucket, Key, Range=None):
        self.calls.append(("get", Key))
        if Key not in self.objects:
            raise _client_error(404, "NoSuchKey", "GetObject")
        data = self.objects[Key][0]
        if Range:
            a, b = Range.split("=")[1].split("-")
            return {"Body": _Body(data[int(a):int(b) + 1]), "ResponseMetadata": {"HTTPStatusCode": 206}}
        return {"Body": _Body(data), "ResponseMetadata": {"HTTPStatusCode": 200}}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + min(MaxKeys, self.page_size)]
        more = start + len(page) < len(keys)
        out = {"Contents": [{"Key": k, "Size": len(self.objects[k][0]),
                             "ETag": '"%s"' % hashlib.md5(self.objects[k][0]).hexdigest()} for k in page],
               "IsTruncated": more}
        if more:
            out["NextContinuationToken"] = str(start + len(page))
        return out


# ---------------------------------------------------------------- synthetic fixtures

def synthetic_csv(eol="\n", quoted=False, days=("2024-04-05", "2024-04-06"), stations=("AAA", "BBB")):
    rows = ["#comment line"] if quoted else []
    rows.append("station,valid,sknt,metar")
    for st in stations:
        for d in days:
            for h in (0, 6, 12, 18):
                remark = f'"{st} 1{h}KT, gusts"' if quoted else f"{st} 1{h}KT"
                rows.append(f"{st},{d} {h:02d}:00,{h % 7},{remark}")
    return (eol.join(rows) + eol).encode("utf-8")


def synthetic_shp(polygons):
    """Bytes of a .shp holding the given polygons (each a list of rings of (lon, lat))."""
    recs = b""
    for n, rings in enumerate(polygons, start=1):
        pts = [p for r in rings for p in r]
        parts, k = [], 0
        for r in rings:
            parts.append(k)
            k += len(r)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        body = struct.pack("<i4d", 5, min(xs), min(ys), max(xs), max(ys)) + struct.pack("<ii", len(rings), len(pts))
        body += struct.pack(f"<{len(parts)}i", *parts) + b"".join(struct.pack("<2d", *p) for p in pts)
        recs += struct.pack(">ii", n, len(body) // 2) + body
    header = struct.pack(">i5ii", 9994, 0, 0, 0, 0, 0, (100 + len(recs)) // 2) + struct.pack("<ii", 1000, 5) + \
        struct.pack("<8d", 0, 0, 0, 0, 0, 0, 0, 0)
    return header + recs


def shp_zip(polygons):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("warnings.shp", synthetic_shp(polygons))
    return buf.getvalue()


FILL32 = np.float32(3.4028235e38)


def synthetic_madis_nc(hour=0, n=12, seed=0, absent=(), types=None, extra=True):
    """A tiny netCDF-3 file shaped like an hourly mesonet file: record dimension ``recNum``, character station ids
    and providers (NUL-padded), float32 values with fill values, ``missing_value`` and a NaN with a payload, a -0.0,
    DD (char) and QCR (int32) flags, records out of key order with repeats and one exact duplicate. ``absent``:
    variables left out; ``types``: {variable: netCDF type code} overrides. Returns (bytes, {name: values written})."""
    from scipy.io import netcdf_file

    rng = np.random.default_rng(seed)
    types = types or {}
    t0 = 1680307200.0 + 3600.0 * hour  # 2023-04-01T00:00Z + hour
    stations = [b"AB1", b"KXYZ", b"C77777", b"D"]
    provs = [b"PROVA", b"MESO-NET-11", b"PROVA", b"X"]
    pick = rng.integers(0, len(stations), n)
    pick[1] = pick[0]  # a repeat of the same station
    data = {
        "stationId": np.array([list(stations[k].ljust(6, b"\0")) for k in pick], np.uint8).view("S1"),
        "dataProvider": np.array([list(provs[k].ljust(11, b"\0")) for k in pick], np.uint8).view("S1"),
        "latitude": np.array([35.0 + 0.5 * k for k in pick], ">f4"),
        "longitude": np.array([-97.0 - 0.25 * k for k in pick], ">f4"),
        "observationTime": (t0 + rng.integers(0, 3600, n)).astype(">f8"),
        "elevation": np.full(n, 300.0, ">f4"),
    }
    data["observationTime"][1] = data["observationTime"][0] + 300.0
    for f in ("windGust", "windSpeed", "windDir", "temperature", "altimeter", "precipAccum"):
        v = (rng.random(n) * 30).astype(">f4")
        v[2], v[3] = FILL32, np.float32(-9999.0)
        data[f] = v
        data[f + "DD"] = np.array([[c] for c in rng.choice(list(b"VSCZ"), n)], np.uint8).view("S1").reshape(n)
        data[f + "QCR"] = rng.integers(0, 4, n).astype(">i4")
    data["windGust"][4] = np.frombuffer(b"\x7f\xc0\x12\x34", ">f4")[0]  # NaN with a payload
    data["windSpeed"][5] = np.float32(-0.0)
    for k in data:  # an exact duplicate of the first record, last
        data[k][-1] = data[k][0]
    fd, path = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    f = netcdf_file(path, "w", version=1)
    f.createDimension("recNum", None)
    f.createDimension("maxStaIdLen", 6)
    f.createDimension("maxStaTypeLen", 11)
    out = {}
    for name, arr in data.items():
        if name in absent or (name == "elevation" and not extra):
            continue
        if name in ("stationId", "dataProvider"):
            v = f.createVariable(name, "c", ("recNum", "maxStaIdLen" if name == "stationId" else "maxStaTypeLen"))
        else:
            tc = types.get(name, {"f4": "f", "f8": "d", "i4": "i", "S1": "c"}[arr.dtype.str[1:]])
            arr = arr.astype(">" + {"f": "f4", "d": "f8", "i": "i4", "h": "i2"}[tc]) if tc != "c" else arr
            v = f.createVariable(name, tc, ("recNum",))
            if tc in "fd" and name not in ("observationTime",):
                v._FillValue = arr.dtype.type(FILL32)
                v.missing_value = arr.dtype.type(-9999.0)
                v.units = b"meter/sec"
        v[:] = arr
        out[name] = arr
    f.close()
    with open(path, "rb") as fh:
        body = fh.read()
    os.unlink(path)
    return body, out


# ---------------------------------------------------------------- checks

def _checks():
    from . import compact, exact, grib, load_config, sparse, stormbox, tile
    from .breaker import Breaker, BreakerTripped, Ledger
    from .manifest import DatasetEntry, DayManifest, validate
    from .store import OverwriteRefused, PrefixError, Store, StoreError, src_key, tile_key, vintage_of

    cfg = load_config()
    day = _dt.date(2024, 4, 5)
    when = _dt.datetime(2024, 4, 6, 4, 12, 33, tzinfo=_dt.timezone.utc)

    def keys():
        v = vintage_of(when)
        assert v == "v20240406T0412Z"
        assert src_key("iem-asos", "v1", day, v, "IA_ASOS.parquet") == \
            "src/iem-asos/v1/2024/2024-04-05/v20240406T0412Z/IA_ASOS.parquet"
        assert tile_key("ru-gust-u8", "v1", day) == "tiles/ru-gust-u8/v1/2024/2024-04-05.pmtiles"
        assert tile_key("ru-gust-u8", "v1", day, 2) == "tiles/ru-gust-u8/v1/2024/2024-04-05~r2.pmtiles"
        s = Store(FakeS3(), "b", write=True)
        for bad in ("other/x", "../src/x", "src//x", "/src/x"):
            try:
                s.put(bad, b"x")
                raise AssertionError("prefix not refused")
            except PrefixError:
                pass

    def store():
        fake = FakeS3()
        s = Store(fake, "b", write=True)
        k = "src/t/v1/2024/2024-04-05/v20240406T0412Z/a.bin"
        assert s.put(k, b"one")["status"] == "written"
        assert s.put(k, b"one")["status"] == "exists-identical"
        try:
            s.put(k, b"two")
            raise AssertionError("overwrite not refused")
        except OverwriteRefused:
            pass
        assert fake.objects[k][0] == b"one"
        ia = "src/t/v1/2024/2024-04-05/v20240406T0412Z/c.bin"
        assert s.put(ia, b"three", storage_class="STANDARD_IA")["status"] == "written"
        assert s.head(ia)["storage_class"] == "STANDARD_IA" and s.head(k)["storage_class"] is None
        try:
            s.put("man/2024/2024-04-05/x.json", b"{}", storage_class="STANDARD_IA")
            refused = False
        except StoreError:
            refused = True
        assert refused, "archive class outside src/ not refused"
        dry = Store(fake, "b", write=False)
        assert dry.put("src/t/v1/2024/2024-04-05/v20240406T0412Z/b.bin", b"x")["status"] == "dry-run"
        assert "src/t/v1/2024/2024-04-05/v20240406T0412Z/b.bin" not in fake.objects

    def manifest():
        s = Store(FakeS3(), "b", write=True)
        e = DatasetEntry("iem-asos", "v1", "v20240406T0412Z")
        e["steps"]["expected"] = e["steps"]["got"] = 1
        e.add_source("https://example.invalid/x", hashlib.sha256(b"r").hexdigest(), 200, 1)
        pr = s.put("src/iem-asos/v1/2024/2024-04-05/v20240406T0412Z/a.parquet", b"data")
        e.add_object(pr)
        assert pr["sha256"] == hashlib.sha256(b"data").hexdigest()
        m = DayManifest(day, {"id": "1", "attempt": "1", "job": "j", "lane": "iem", "git_sha": "x",
                              "lock_sha256": None, "runner_image": "local", "started_utc": "x", "ended_utc": None})
        m.add(e)
        e.finish_status()
        assert validate(m.to_doc()) == []
        e["checks"]["max_gust"] = 1
        assert validate(m.to_doc()) != []

    def compaction():
        for data in (synthetic_csv(), synthetic_csv("\r\n", quoted=True)):
            t = compact.csv_to_table(data)
            assert compact.table_to_bytes(t) == data
            parts = [compact.table_to_parquet(x) for x in compact.slice_by_day(t, ["valid"]).values()]
            assert compact.roundtrip_ok(parts, hashlib.sha256(data).hexdigest())

    def sparse_cube():
        rng = np.random.default_rng(1)
        shape = (40, 50)
        pack = {"template": 3, "R": 0.09325958788394928, "E": -4, "D": 0, "nbits": 9, "bitmap": False}
        cube = sparse.SparseCube(shape, 3, 35 / sparse.MPH_PER_MS, 10, [5, 77], [6])
        steps = [grib.ints_to_values(rng.integers(0, 512, shape), pack) for _ in range(3)]
        steps[1][3, 4] = np.nan
        for i, v in enumerate(steps):
            cube.add_step(i, v, pack)
        data, rows = cube.to_parquet({"t": 1})
        assert sparse.decode_check(data, cube) and rows > 0
        assert cube.max_quant_err == 0.0 and cube.exact_cells == cube.total_cells
        d = sparse.dense_step(data, 0)
        kept = np.isfinite(d)
        assert grib.same_bits(d[kept], steps[0][kept])
        lossy = sparse.SparseCube(shape, 1, 0.0, 1, [])
        lossy.add_step(0, steps[0], None)  # 1/16 m/s values without their packing cannot be exact
        assert not sparse.decode_check(lossy.to_parquet({})[0], lossy)

    def exact_grid():
        rng = np.random.default_rng(4)
        shape = (150, 140)
        pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
        with tempfile.TemporaryDirectory() as tmp:
            cube = exact.ExactCube(shape, 3, tmp, 64, "dxy", 3, "auto", 2)
            vals = [grib.ints_to_values(rng.integers(0, 4096, shape), pack) for _ in range(2)]
            assert cube.add_step(0, "a", vals[0], pack, b"\x00" * 200000)["mode"] == "ints"
            assert cube.add_step(2, "c", vals[1], pack, b"short")["mode"] == "grib"
            data = cube.to_bytes({"layer": "x"})
            cube.close()
        ok, rep = exact.verify(data)
        assert ok and rep["checked"] == 2
        r = exact.ExactReader.from_bytes(data)
        assert grib.same_bits(r.values(0), vals[0]) and r.step(1)["mode"] == "missing" and r.source(2) == b"short"
        bad = bytearray(data)
        bad[-200] ^= 1
        assert not exact.verify(bytes(bad))[0]

    def exact_v2():
        # med prediction with zigzag residuals, and the reflectivity palette with its special codes
        i, j = np.mgrid[0:150, 0:140]
        refl = {"template": 41, "R": -9990.0, "E": 0, "D": 1, "nbits": 16, "bitmap": False}
        x = np.full((150, 140), 9000, np.uint32)                      # -99 no echo
        x[:, :40] = 0                                                 # -999 outside coverage
        m = (i - 70) ** 2 + (j - 80) ** 2 < 1600
        x[m] = (9990 + 5 * ((i + 2 * j) % 150))[m]
        ramp = (60000 - 3 * i - 5 * j).astype(np.uint32)             # negative residuals everywhere
        with tempfile.TemporaryDirectory() as tmp:
            out = {}
            for name, pred, pal, ints, pack in (("refl", "med", "auto", x, refl), ("ramp", "med", "off", ramp, refl),
                                                ("ramp-dxy", "dxy", "off", ramp, refl)):
                cube = exact.ExactCube(ints.shape, 1, tmp, 64, pred, 3, "ints", 2, pal)
                rec = cube.add_step(0, "a", grib.ints_to_values(ints, pack), pack, b"src")
                out[name] = (cube.to_bytes({"layer": name}), rec)
                cube.close()
        data, rec = out["refl"]
        assert data[:8] == b"XGRID002" and rec["dtype"] == "u1" and rec["palette"]["count"] == len(np.unique(x))
        for name, ints in (("refl", x), ("ramp", ramp), ("ramp-dxy", ramp)):
            assert exact.verify(out[name][0])[0], name
            assert np.array_equal(exact.ExactReader.from_bytes(out[name][0]).ints(0), ints), name
        bad = bytearray(data)
        bad[-20] ^= 4
        assert not exact.verify(bytes(bad))[0]

    def parallel():
        # a step prepared in a worker process is the same bytes as one prepared here; decode-back split over the
        # workers gives the same verdict
        from . import par

        rng = np.random.default_rng(6)
        shape = (200, 150)
        pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
        vals = [grib.ints_to_values(rng.integers(0, 4096, shape), pack) for _ in range(3)]
        if par.pool() is None:
            return "skipped (FEEDER_PROCS: no worker processes)"
        with tempfile.TemporaryDirectory() as tmp:
            out = []
            for remote in (False, True):
                cube = exact.ExactCube(shape, 3, tmp, 64, "dxy", 3, "auto", 2)
                for i, v in enumerate(vals):
                    src = b"\x00" * 200000
                    if remote:
                        f = par.pool().submit(exact.prepare_step, v, pack, src, "none", workers=1, **cube.params())
                        cube.commit_step(i, str(i), f.result())
                    else:
                        cube.add_step(i, str(i), v, pack, src)
                out.append(cube.to_bytes({"layer": "x"}))
                cube.close()
            assert out[0] == out[1]
            assert exact.verify(out[1], submit=par.submit, parts=3, staging_dir=tmp) == exact.verify(out[0])
        par.shutdown()

    def mesh_codes():
        enc = tile.Encoding(cfg["encodings"]["mesh05"])
        _, ov = enc.encode(np.array([-3.0, -1.0, 0.8, 50.0, -999.0, -99.0]))
        assert not ov.any()

    def tiles():
        assert [tile.zxy_to_tileid(1, x, y) for x, y in ((0, 0), (0, 1), (1, 1), (1, 0))] == [1, 2, 3, 4]
        assert tile.zxy_to_tileid(0, 0, 0) == 0
        enc = tile.Encoding(cfg["encodings"]["mph05"])
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(2)
            cube = tile.TileCube((300, 260), 4, enc, tmp, 128)
            for i in range(4):
                v = rng.uniform(0, 40, (300, 260))
                v[0, 0] = 90.0  # 201 mph: overflow sidecar
                v[1, 1] = np.nan
                cube.add_step(i, v)
            assert cube.decode_back_ok
            path = tmp + "/t.pmtiles"
            cube.write(path, {"layer": "x"})
            assert cube.verify_file(path)
            assert cube.sidecar_parquet() is not None
            with open(path, "rb") as fh:
                r = tile.PMTilesReader.from_bytes(fh.read())
            assert r.metadata()["grid_shape"] == [300, 260]
            cube.close()

    def breaker():
        s = Store(FakeS3(), "b", write=True)
        b = Breaker(cfg["breaker"], cfg["cuts"], "noaa", Ledger(s, "r1", "j1"))
        b.after_day(day, 100, False, verified=100, verify_failed=1)
        try:
            b.check_before_day()
            raise AssertionError("integrity rule did not trip")
        except BreakerTripped:
            pass

    def boxes():
        poly = [[(-97.1, 35.1), (-96.9, 35.1), (-96.9, 35.3), (-97.1, 35.3)]]
        got = stormbox.polygon_rule(stormbox.polygons_from_zip(shp_zip([poly, poly, [[(-80, 40), (-79.9, 40),
                                                                                     (-79.9, 40.1)]]])))
        # the polygons straddle 97 W, so both neighbouring boxes are touched; the single third polygon is not enough
        assert got == sorted([stormbox.box_id(13, 5), stormbox.box_id(14, 5)]), got

    def madis_table():
        from . import madis

        spec = cfg["madis"]
        hours = [madis.parse_hour(synthetic_madis_nc(hour=h, seed=h)[0], spec, index=h) for h in (0, 1)]
        hours.append(madis.parse_hour(synthetic_madis_nc(hour=2, seed=2, absent=("precipAccumDD",))[0], spec, 2))
        blob = madis.to_parquet(spec, madis.build_table(spec, hours, {"schema": madis.SCHEMA}))
        ok, why = madis.decode_back_ok(spec, blob, hours)
        assert ok, why
        hours[1].cols["windGust"][0] = np.float32(1.5)  # a changed value must be caught
        assert not madis.decode_back_ok(spec, blob, hours)[0]

    def grib_lib():
        import importlib

        try:
            importlib.import_module("pygrib")
        except ImportError:
            return "skipped (pygrib not installed)"

    return [("keys", keys), ("store", store), ("manifest", manifest), ("compact", compaction),
            ("sparse", sparse_cube), ("exact", exact_grid), ("exact-v2", exact_v2), ("parallel", parallel), ("mesh-codes", mesh_codes), ("tiles", tiles), ("breaker", breaker), ("boxes", boxes), ("madis", madis_table), ("pygrib", grib_lib)]


def main():
    bad = 0
    for name, fn in _checks():
        try:
            note = fn()
            print(f"selftest {name}: pass" + (f" ({note})" if note else ""), flush=True)
        except Exception as err:  # report and continue
            bad += 1
            print(f"selftest {name}: FAIL {type(err).__name__}: {err}", flush=True)
    print(f"selftest: {'FAIL' if bad else 'pass'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
