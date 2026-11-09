"""The parallel gridded lane stores the same bytes as the serial one.

- the in-place integer / value conversions and the copy-free checksums equal the expression forms they replace;
- the ordered pipelines keep item order, bound the work in flight and raise at the failing item's turn;
- a fixture day (URMA with station cells, sparse cube and box flags; a listed reflectivity-like group with a missing
  step, a failed fetch and a step kept as its source bytes; MESH; ProbSevere on its side thread) gives the same
  objects, byte for byte, and the same manifest entries with worker processes as without them, and the objects'
  sha256 equal those the serial code stored before the parallel lane existed (GOLDEN).
"""

import copy
import gzip
import hashlib
import threading
import time

import numpy as np
import pytest

from feeder import exact, grib, par
from feeder.manifest import parse
from feeder.run import lane_noaa

from conftest import DAY, NX, NY, add_urma, fake_message, make_ctx

# sha256 of every object this fixture day stores, as written by the serial code before the parallel lane (git
# 238dac3); file name -> sha256. The .xgrid objects are exact-grid format 2 (golden bytes of the new encoding,
# pinned zstandard 0.25.0); their format 1 sha256 were refl-x ddb09ce6..., urma-gust-x e7b0b38a...,
# urma-wdir-x 32ca77b3..., urma-wind-x e66b4975....
GOLDEN = {
    "boxflags-mesh-20240405.json": "ccba38decb55013185e876f78937b6dd58d4b46249824a2c4c58cf7e18558231",
    "boxflags-urma-20240405.json": "d9ddd2019e6163d7693d4e37fb7664b10136a027bd2678388548fdb45bfcbaf9",
    "mrms-mesh-raw-20240405.tar": "7ff712227bc4a63d1b15c0b51e68a0d438a28e2456ebe5595e50972b3ee12da6",
    "probsevere-json-20240405.tar.zst": "a7567e872ef459082f837fdeceed66c66a51c288564b6d1617c4577d34ee1872",
    "refl-x-20240405.xgrid": "3c7589cb2341795fedcc76d6fbd49f0bda364d5922223566571885d0803a4ca6",
    "urma-gust-ex-20240405.parquet": "e154c3ca9c725d59088a06ab5eb84925d463ea9d6da6250253b689c565f4d1aa",
    "urma-gust-x-20240405.xgrid": "9875147cb71234441fe747ae06045e7f5410970450b80931548f10f1d3888331",
    "urma-wdir-x-20240405.xgrid": "0217b6702b099664cca74c66a359c2518405acaa2c5afde6cb6b0a832da40146",
    "urma-wind-x-20240405.xgrid": "0fd5c02fe20bbd35804e09f26030f3aeac86ea5937d514ead5d6dd2b609835c6",
}


# ---------------------------------------------------------------- conversions and checksums

def _old_values_to_ints(values, packing):
    """The expression form used before (reference)."""
    if not packing or packing.get("bitmap"):
        return None
    v = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return None
    s = grib.power(2.0, int(packing["E"]))
    d = grib.power(10.0, -int(packing["D"]))
    x = np.rint((v / d - float(packing["R"])) / s)
    if x.size and (x.min() < 0 or x.max() > 0xFFFFFFFF):
        return None
    ints = x.astype(np.uint32)
    if not grib.same_bits(_old_ints_to_values(ints, packing), v):
        return None
    return ints


def _old_ints_to_values(ints, packing):
    s = grib.power(2.0, int(packing["E"]))
    d = grib.power(10.0, -int(packing["D"]))
    return (np.asarray(ints, dtype=np.float64) * s + float(packing["R"])) * d


PACKS = [
    {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12},
    {"template": 0, "R": 2131.0, "E": 0, "D": 1, "nbits": 12},
    {"template": 3, "R": 0.09325958788394928, "E": -4, "D": 0, "nbits": 9},
    {"template": 41, "R": -9990.0, "E": 0, "D": 1, "nbits": 16},
    {"template": 0, "R": -3.3e-5, "E": 3, "D": -2, "nbits": 20},
    {"template": 0, "R": 1.7976e300, "E": -12, "D": 5, "nbits": 32},
]


@pytest.mark.parametrize("pack", PACKS)
def test_in_place_conversions_equal_the_expression_forms(pack):
    rng = np.random.default_rng(11)
    pk = {**pack, "bitmap": False}
    x = rng.integers(0, 2 ** min(pack["nbits"], 31), size=(97, 131)).astype(np.uint32)
    v = _old_ints_to_values(x, pk)
    assert grib.same_bits(grib.ints_to_values(x, pk), v)
    assert grib.same_bits(grib.ints_to_values(x.astype(np.int64), pk), _old_ints_to_values(x.astype(np.int64), pk))
    xf = x.astype(np.float64)
    before = xf.copy()
    grib.ints_to_values(xf, pk)
    assert grib.same_bits(xf, before)  # never writes into its input
    for w in (v, v + 1e-7, v * (1 + 1e-15), np.where(x % 7 == 0, v + 0.5, v), -v, v + np.nextafter(0, 1)):
        a, b = grib.values_to_ints(w, pk), _old_values_to_ints(w, pk)
        assert (a is None) == (b is None)
        if a is not None:
            assert a.dtype == b.dtype and np.array_equal(a, b)
    assert grib.values_to_ints(np.zeros((0, 3)), pk) is not None


def test_copy_free_checksums_equal_tobytes():
    rng = np.random.default_rng(3)
    v = rng.normal(size=(64, 50))
    v[0, 0], v[1, 1], v[2, 2] = np.nan, -0.0, np.inf
    x = rng.integers(0, 2 ** 31, size=(64, 50)).astype(np.uint32)
    for arr in (v, v[:, ::2], v.T, v.astype(">f8")):
        assert exact.values_sha256(arr) == \
            hashlib.sha256(np.ascontiguousarray(arr, dtype="<f8").tobytes()).hexdigest()
    for arr in (x, x[::3], x.T, x.astype(np.uint16), x.astype(">u4")):
        assert exact.ints_sha256(arr) == hashlib.sha256(np.ascontiguousarray(arr, dtype="<u4").tobytes()).hexdigest()


def test_prepared_steps_and_parallel_verify_equal_serial(tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDER_PROCS", "3")
    rng = np.random.default_rng(8)
    pack = {"template": 0, "R": 0.0, "E": 0, "D": 2, "nbits": 12, "bitmap": False}
    shape = (300, 270)
    steps = [grib.ints_to_values(rng.integers(0, 4096, shape).astype(np.uint32), pack) for _ in range(5)]
    steps[3] = steps[3] + 1e-9  # not reproducible from integers: kept as source bytes
    srcs = [b"src-%d" % i * 200000 for i in range(5)]
    srcs[2] = b"tiny"  # source smaller than the integers: kept as is

    def build(prepare_remote):
        cube = exact.ExactCube(shape, 7, str(tmp_path), 128, "dxy", 3, "auto", 2)
        for i, (v, s) in enumerate(zip(steps, srcs)):
            if prepare_remote:
                f = par.pool().submit(exact.prepare_step, v, pack, s, "none", workers=1, **cube.params())
                cube.commit_step(i + 1, f"t{i}", f.result())
            else:
                cube.add_step(i + 1, f"t{i}", v, pack, s)
        data = cube.to_bytes({"k": "x"})
        cube.close()
        return data

    serial, parallel = build(False), build(True)
    assert serial == parallel
    want = exact.verify(serial)
    assert want[0] and want[1]["checked"] == 5
    assert exact.verify(serial, submit=par.submit, parts=4, staging_dir=str(tmp_path)) == want
    bad = bytearray(serial)
    bad[-200] ^= 1
    want_bad = exact.verify(bytes(bad))
    assert not want_bad[0]
    assert exact.verify(bytes(bad), submit=par.submit, parts=4, staging_dir=str(tmp_path)) == want_bad
    assert not list(tmp_path.glob("*.xgv"))  # the verify copy is removed


# ---------------------------------------------------------------- pipelines

def test_ordered_keeps_order_bounds_work_and_raises_in_turn():
    live, peak, lock = [0], [0], threading.Lock()

    def job(x):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.002 * ((x * 7) % 5))
        with lock:
            live[0] -= 1
        if x == 13:
            raise ValueError("thirteen")
        return x * x

    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(8) as tp:
        got = []
        with pytest.raises(ValueError):
            for item, r in par.ordered(lambda x: tp.submit(job, x) if x % 4 else ("pass", x), range(40), 3):
                got.append((item, r))
        assert [i for i, _ in got] == list(range(13))
        assert all(r == (("pass", i) if i % 4 == 0 else i * i) for i, r in got)
        assert peak[0] <= 3
        assert list(par.prefetch(tp, lambda x: x + 1, range(25), 4)) == list(range(1, 26))
    # inline work starts one item at a time, just before it is consumed
    started = []
    for item, r in par.ordered(lambda x: started.append(x) or par.inline(lambda: x), range(5), 1):
        assert started[-1] == item


# ---------------------------------------------------------------- a fixture day, serial vs worker processes

RSHAPE = (260, 300)  # several 128 x 128 blocks, edge blocks smaller
REFL_PACK = {"R": -9990.0, "E": 0, "D": 1}


def _refl_values(k):
    rng = np.random.default_rng(100 + k)
    v = np.full(RSHAPE, -99.0)
    v[:40, :] = -999.0
    y, x = 60 + 5 * k, 80 + 7 * k
    v[y:y + 70, x:x + 90] = np.round(rng.uniform(-20, 65, (70, 90)) * 2) / 2
    return v


def add_refl(web, day=DAY):
    """A listed group: a 10-minute cadence over 3 hours, one listed file whose fetch fails (500), one step with
    no file, and one step whose values the integers cannot rebuild (kept as the source bytes)."""
    ymd = f"{day:%Y%m%d}"
    keys, k = [], 0
    for h in range(3):
        for m in range(0, 60, 10):
            if (h, m) == (1, 20):
                continue  # nothing listed: missing step
            name = f"CONUS/MergedReflectivityQCComposite_00.50/{ymd}/" \
                   f"MRMS_MergedReflectivityQCComposite_00.50_{ymd}-{h:02d}{m:02d}36.grib2.gz"
            keys.append(name)
            url = f"https://noaa-mrms-pds.s3.amazonaws.com/{name}"
            if (h, m) == (2, 30):
                web.add(url, b"", status=500)
                continue
            v = _refl_values(k)
            if (h, m) == (0, 50):
                msg = fake_message((209, 0, 16), v + 1e-9)  # unpacked float values: no integers
            else:
                msg = fake_message((209, 0, 16), v, REFL_PACK)
            web.add(url, gzip.compress(msg, mtime=0))
            k += 1
    listing = ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">' +
               "".join(f"<Contents><Key>{n}</Key><ETag>\"e\"</ETag><Size>1</Size></Contents>" for n in keys) +
               "<IsTruncated>false</IsTruncated></ListBucketResult>").encode()

    def fn(url, p):
        if p.netloc == "noaa-mrms-pds.s3.amazonaws.com" and p.path == "/" and "MergedReflectivity" in p.query:
            return 200, {}, listing
        return None

    web.route(fn)


def add_mesh_probsevere(web, day=DAY):
    def fn(url, p):
        if p.netloc != "noaa-mrms-pds.s3.amazonaws.com":
            return None
        if "MESH_Max" in p.path:
            hour = int(p.path.rsplit("-", 1)[1][:2])
            v = np.zeros((NY, NX))
            v[0:6, 0:6] = 30.0 + hour
            return 200, {}, gzip.compress(fake_message((209, 3, 30), v), mtime=0)
        if p.path == "/" and "ProbSevere" in p.query:
            keys = "".join(f"<Contents><Key>ProbSevere/{day:%Y%m%d}/MRMS_PROBSEVERE_{day:%Y%m%d}_"
                           f"{h:02d}0040.json</Key><ETag>\"e{h}\"</ETag><Size>10</Size></Contents>" for h in range(5))
            return 200, {}, ('<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">' + keys +
                             "<IsTruncated>false</IsTruncated></ListBucketResult>").encode()
        if p.path.endswith(".json"):
            return 200, {}, ('{"features": [%d]}' % len(p.path)).encode() * 50
        return None

    web.route(fn)


def fixture_day(cfg, web, clock, s3, tmp_path, run_id):
    """Runs the fixture day; returns ({stored file name: bytes}, [manifest dataset entries])."""
    add_urma(web, missing_hours=(5,))
    add_refl(web)
    add_mesh_probsevere(web)
    cfg = copy.deepcopy(cfg)
    for g in cfg["noaa"]["groups"]:
        if g["id"] == "mrms-refl":
            g["outputs"] = [o for o in g["outputs"] if o["kind"] == "exact"]
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma", "mrms-refl", "mrms-mesh", "probsevere"],
                   run_id=run_id, job=f"job-{run_id}")
    lane_noaa(ctx, DAY, DAY, pilot=True)
    ctx.close()
    objs = {k.rsplit("/", 1)[1]: v[0] for k, v in s3.objects.items()
            if k.startswith(("src/", "tiles/")) and not k.startswith("src/iem-stations/")}  # not the seeded input
    mans = [parse(v[0]) for k, v in sorted(s3.objects.items()) if k.startswith("man/")]
    assert len(mans) == 1
    return objs, sorted(mans[0]["datasets"], key=lambda d: d["id"])


@pytest.fixture
def fresh(cfg, clock, tmp_path):
    from conftest import FakeWeb
    from feeder.selftest import FakeS3

    from feeder import noaa

    def make(procs, monkeypatch):
        monkeypatch.setenv("FEEDER_PROCS", str(procs))
        par.shutdown()
        remote, init = [], noaa._Steps.__init__

        def spy(self, *a):
            init(self, *a)
            remote.append(self.remote)

        monkeypatch.setattr(noaa._Steps, "__init__", spy)
        s3 = FakeS3()
        stage = tmp_path / f"p{procs}"
        stage.mkdir()
        out = fixture_day(cfg, FakeWeb(clock), clock, s3, stage, run_id="500")
        assert remote == [procs > 1] * 2  # urma and the reflectivity group: in worker processes, or inline
        return out

    yield make
    par.shutdown()


def test_fixture_day_same_bytes_serial_and_workers(fresh, monkeypatch):
    serial_objs, serial_ds = fresh(1, monkeypatch)
    assert par.pool() is None
    workers_objs, workers_ds = fresh(3, monkeypatch)
    assert sorted(serial_objs) == sorted(GOLDEN)
    assert workers_objs == serial_objs  # every stored object, byte for byte
    assert workers_ds == serial_ds      # every manifest entry: sources, missing steps, checks, objects and sha256
    got = {k: hashlib.sha256(v).hexdigest() for k, v in serial_objs.items()}
    assert got == GOLDEN  # the bytes the serial code stored before the parallel lane
    by = {d["id"]: d for d in workers_ds}
    rx = by["refl-x"]
    assert rx["checks"]["decode_back"] == "pass"
    assert rx["checks"]["exact"] == {"steps_ints": 15, "steps_grib": 1, "steps_missing": 144 - 16}
    assert {m["http"] for m in rx["steps"]["missing"]} == {404, 500}
    assert by["urma-gust-x"]["checks"]["exact"]["steps_missing"] == 1
    assert by["urma-gust-ex"]["checks"]["station_points"] == 3
    assert by["probsevere-json"]["checks"]["members"] == 5


def test_peak_memory_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("FEEDER_PROCS", "2")
    par.shutdown()
    par.pool().submit(sum, [1, 2]).result()
    m = par.log_memory("x")
    assert m["main_mb"] > 0 and 1 <= m["workers"] <= 2 and m["workers_mb"] > 0  # workers start on demand
    assert "peak-mem x main_mb=" in capsys.readouterr().out
    par.shutdown()


def _die():
    import os

    os._exit(3)


def test_a_broken_worker_pool_is_replaced(monkeypatch):
    import concurrent.futures.process as cfp

    monkeypatch.setenv("FEEDER_PROCS", "2")
    par.shutdown()
    with pytest.raises(cfp.BrokenProcessPool):
        par.pool().submit(_die).result()
    assert par.pool().submit(sum, [2, 3]).result() == 5
    par.shutdown()


def test_overlap_group_runs_beside_the_others_and_its_errors_surface(cfg, web, clock, s3, tmp_path, monkeypatch):
    from feeder import noaa
    from feeder.run import Book, run_noaa_day
    from feeder.store import PrefixError

    assert next(g for g in cfg["noaa"]["groups"] if g["id"] == "probsevere").get("overlap") is True
    ctx = make_ctx(cfg, web, clock, s3, tmp_path, groups=["urma", "probsevere"])
    threads = {}

    def fake(ctx_, g, day, todo, supersedes=None):
        threads[g["id"]] = threading.current_thread().name
        if g["id"] == "probsevere" and fail[0]:
            raise PrefixError("x")
        res = noaa.GroupResult()
        res.reasons.append(g["id"])
        return res

    fail = [False]
    monkeypatch.setattr(noaa, "process_group", fake)
    book = Book(ctx)
    run_noaa_day(ctx, book, DAY)
    assert threads["urma"] == threading.current_thread().name != threads["probsevere"]
    fail[0] = True
    with pytest.raises(PrefixError):
        run_noaa_day(ctx, Book(ctx), DAY)
