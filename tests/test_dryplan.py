import json

from feeder import dryplan, grib
from feeder.net import Http

from conftest import DAY, add_iem, add_urma, fake_decode


def test_dry_plan_offline(cfg, web, clock, monkeypatch, capsys):
    add_urma(web)
    add_iem(web)
    cfg["noaa"]["groups"] = [g for g in cfg["noaa"]["groups"] if g["id"] == "urma"]
    monkeypatch.setattr(dryplan, "load_config", lambda: cfg)
    monkeypatch.setattr(dryplan, "Http", lambda c: Http(c, transport=web, sleep=clock.sleep, clock=clock))
    monkeypatch.setattr(grib, "decode", fake_decode)
    code = dryplan.main({"IN_DAY": DAY.isoformat(), "SAMPLE_HOUR": "5"})
    out = capsys.readouterr().out
    assert code == 0
    doc = json.loads(next(line for line in out.splitlines() if line.startswith("{")))
    o = doc["outputs"]
    assert set(o) == {"urma-gust-x", "urma-wind-x", "urma-wdir-x", "urma-gust-ex", "boxflags-urma"}
    for k in ("urma-gust-x", "urma-wind-x", "urma-wdir-x"):
        assert o[k]["sample"]["decode_back"] == "pass" and o[k]["steps_found"] == 24 and o[k]["stored_bytes"] > 0
    sp = o["urma-gust-ex"]["sample"]
    assert sp["decode_back"] == "pass" and sp["exact_cells"] == sp["total_cells"] and sp["station_points"] == 6
    s = doc["summary"]
    assert s["objects_per_day"] == 6 and s["cap_gb"] == 2500.0 and s["projected_total_gb_to_date"] > 310
    # only .idx files and one sampled hour of messages were read from the gridded source
    # (plus a 1-byte read for the file length where the wanted message is the last one in the file)
    grid_reads = [x for x in web.log if "noaa-urma-pds" in x["url"] and not x["url"].endswith(".idx")
                  and x["headers"].get("Range") != "bytes=0-0"]
    assert len(grid_reads) == 3
