"""Tiny live smoke test (opt-in: FEEDER_LIVE=1; files go to FEEDER_SMOKE_DIR).

Downloads: one URMA .idx (checks the wind field patterns), one byte range (< 5 MB, one whole GRIB2 message) and
one 1-station 1-day observation CSV. Total well under 50 MB."""

import datetime as _dt
import hashlib
import os

import numpy as np
import pytest

from feeder import compact, grib, iem, load_config, noaa
from feeder.net import Http

pytestmark = [pytest.mark.live,
              pytest.mark.skipif(os.environ.get("FEEDER_LIVE") != "1", reason="live smoke test is opt-in")]

LIMIT = 50 * 1024 * 1024


@pytest.fixture(scope="module")
def http():
    return Http(load_config())


@pytest.fixture(scope="module")
def outdir():
    d = os.environ.get("FEEDER_SMOKE_DIR") or os.path.join(os.getcwd(), "out", "smoke")
    os.makedirs(d, exist_ok=True)
    return d


def _save(outdir, name, data):
    with open(os.path.join(outdir, name), "wb") as fh:
        fh.write(data)


def test_urma_idx_range_decode(http, outdir):
    cfg = load_config()
    group = next(g for g in cfg["noaa"]["groups"] if g["id"] == "urma")
    when = _dt.datetime(2024, 7, 16, 12, tzinfo=_dt.timezone.utc)
    url = noaa.nodd_url(group["bucket"], noaa.fill(group["key"], when))
    ir = http.get(url + group["idx_suffix"])
    assert ir.status == 200
    _save(outdir, "urma.t12z.idx", ir.body)
    entries = noaa.parse_idx(ir.body.decode("ascii"))
    for name in ("gust", "wind", "wdir"):
        assert noaa.find_entry(entries, group["fields"][name]["idx"]) is not None, name
    # URMA wind messages use simple packing and are ~5.6 MB each, above the smoke budget for one read; the last
    # significant-wave-height message (~1.8 MB) of the same file exercises the same path: .idx byte range,
    # GRIB2 header parse, pygrib decode with parameter-id check, grid locator on the 2.5-km grid.
    e = noaa.find_entry(entries, "HTSGW:surface:anl")
    assert e is not None and e.end is not None
    r = http.get(url, rng=(e.offset, e.end))
    assert r.status == 206 and r.etag
    size = len(r.body)
    assert 0 < size < 5_000_000
    _save(outdir, "urma.t12z.htsgw.grib2", r.body)
    total = len(ir.body) + size
    field = {"grib": [10, 0, 3]}
    h = grib.parse_header(r.body[:4096])
    assert h["length"] == size and grib.matches(h, field)
    fld = grib.decode(r.body, field)
    assert fld.ids == (10, 0, 3)
    ny, nx = fld.values.shape
    assert ny * nx > 2_000_000
    assert np.isfinite(fld.values).any()
    cell = fld.grid.locate(41.534, -93.653)  # a point in Iowa
    assert cell is not None
    lats, lons = fld.grid.latlons()
    assert abs(lats.ravel()[cell] - 41.534) < 0.03 and abs(lons.ravel()[cell] + 93.653) < 0.03
    print("smoke urma:", f"idx_bytes={len(ir.body)}", f"range_bytes={size}", f"grid={ny}x{nx}",
          f"sha256={hashlib.sha256(r.body).hexdigest()}")
    assert total < LIMIT


def test_iem_asos_one_station_one_day(http, outdir):
    cfg = load_config()
    ds = next(d for d in cfg["iem"]["datasets"] if d["id"] == "iem-asos")
    t0 = _dt.datetime(2024, 7, 16, tzinfo=_dt.timezone.utc)
    params = [("station", "DSM")] + [tuple(p) for p in ds["params"]] + \
        iem.time_params(ds["time_params"], t0, t0 + _dt.timedelta(days=1))
    url = iem.build_url(cfg["iem"]["base"], ds["path"], params)
    r = http.get(url)
    assert r.status == 200 and len(r.body) < LIMIT
    _save(outdir, "asos-DSM-20240716.csv", r.body)
    t = compact.csv_to_table(r.body)
    assert t.num_rows > 20
    assert compact.table_to_bytes(t) == r.body
    slices = compact.slice_by_day(t, ds["time_column"])
    assert set(slices) == {"2024-07-16"}
    p = compact.table_to_parquet(slices["2024-07-16"])
    _save(outdir, "asos-DSM-20240716.parquet", p)
    assert compact.roundtrip_ok([p], r.sha256)
    print("smoke iem:", f"bytes={len(r.body)}", f"rows={t.num_rows}", f"parquet_bytes={len(p)}",
          f"columns={len(t.column_names) - 1}")
