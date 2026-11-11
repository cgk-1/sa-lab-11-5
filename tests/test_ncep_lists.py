from feeder import compact, other
from feeder.manifest import parse
from feeder.run import lane_iem

from conftest import DAY, add_iem, add_other, make_ctx

IDS = ("ncep-rtma2p5-lists", "ncep-rtma2p5-ru-lists", "ncep-urma2p5-lists")


def _entries(s3, ds_id):
    out = []
    for k, v in sorted(s3.objects.items()):
        if k.startswith("man/"):
            out += [d for d in parse(v[0])["datasets"] if d["id"] == ds_id]
    return out


def test_latest_version_directory_is_listed():
    ds = {"index_from": {"parent": "P/", "dir": r"rtma\.v(\d+)\.(\d+)\.(\d+)/", "sub": "fix/rtma2p5/"}}
    body = b'<a href="rtma.v2.10.9/"></a><a href="rtma.v2.10.12/"></a><a href="rtma.v2.9.30/"></a>'
    assert other.latest_index(body, ds) == "P/rtma.v2.10.12/fix/rtma2p5/"
    assert other.latest_index(b"nothing", ds) is None


def test_lists_kept_by_content_only_once(cfg, web, clock, s3, tmp_path, monkeypatch):
    monkeypatch.setenv("CONTACT_EMAIL", "ops@example.invalid")
    add_iem(web)
    add_other(web)
    lane_iem(make_ctx(cfg, web, clock, s3, tmp_path, lane="pilot", job="p1"), DAY, DAY, "pilot")
    for ds_id in IDS:
        keys = [k for k in s3.objects if k.startswith(f"src/{ds_id}/v1/")]
        # the two list files, never the binary; text kept zstd-compressed and byte-exact
        assert len(keys) == 2, (ds_id, keys)
        assert not any("berror" in k for k in keys)
        for k in keys:
            assert compact.unzstd(s3.objects[k][0]).startswith(b"list ")
        (e,) = _entries(s3, ds_id)
        assert all("/rtma.v2.10.12/" in s["url"] or "/urma.v2.10.10/" in s["url"] for s in e["sources"])
    before = set(s3.objects)
    lane_iem(make_ctx(cfg, web, clock, s3, tmp_path, lane="pilot", job="p2"), DAY, DAY, "pilot")
    for ds_id in IDS:
        assert {k for k in s3.objects if k.startswith(f"src/{ds_id}/")} == {k for k in before if k.startswith(f"src/{ds_id}/")}
