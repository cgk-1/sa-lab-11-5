import base64
import datetime as _dt
import hashlib

import pytest

from feeder.selftest import FakeS3
from feeder.store import (OverwriteRefused, PrefixError, Store, check_key, idx_key, man_key, run_key, src_key,
                          tile_key, vintage_of, vrf_key)

DAY = _dt.date(2023, 7, 1)
V = "v20261030T0412Z"


def test_key_layout():
    assert vintage_of(_dt.datetime(2026, 10, 30, 4, 12, 59, tzinfo=_dt.timezone.utc)) == V
    assert src_key("urma-gust-ex", "v1", DAY, V, "urma-gust-ex-20230701.parquet") == \
        "src/urma-gust-ex/v1/2023/2023-07-01/v20261030T0412Z/urma-gust-ex-20230701.parquet"
    assert tile_key("refl-u8", "v1-c2", DAY) == "tiles/refl-u8/v1-c2/2023/2023-07-01.pmtiles"
    assert tile_key("refl-u8", "v1", DAY, 1) == "tiles/refl-u8/v1/2023/2023-07-01~r1.pmtiles"
    assert man_key(DAY, "123", "2", "noaa-2023-07-01") == "man/2023/2023-07-01/123-2-noaa-2023-07-01.json"
    assert vrf_key(DAY, "123", "1", "verify-x") == "vrf/2023-07-01/123-1-verify-x.json"
    assert idx_key("v1", DAY, "123", "1") == "idx/v1/2023-07-01/123-1.json"
    assert run_key("123", "noaa-2023-07-01-1", 7) == "run/123/noaa-2023-07-01-1/000007.json"


@pytest.mark.parametrize("bad", ["tmp/x", "src", "../src/a", "src/../x", "src//a", "/src/a", "src/a b",
                                 "wl/latest.json", "src/" + "a" * 1000])
def test_prefix_refused(bad):
    with pytest.raises(PrefixError):
        check_key(bad)
    s = Store(FakeS3(), "b", write=True)
    with pytest.raises(PrefixError):
        s.put(bad, b"x")


@pytest.mark.parametrize("args", [("Bad_Name", "v1"), ("ok", "1"), ("ok", "v1x")])
def test_bad_names(args):
    with pytest.raises(PrefixError):
        src_key(args[0], args[1], DAY, V, "f.bin")
    with pytest.raises(PrefixError):
        src_key("ok", "v1", DAY, "v2026", "f.bin")
    with pytest.raises(PrefixError):
        src_key("ok", "v1", DAY, V, "a/b.bin")


def test_put_sends_conditional_and_md5():
    fake = FakeS3()
    calls = []
    orig = fake.put_object

    def spy(**kw):
        calls.append(kw)
        return orig(**kw)

    fake.put_object = spy
    s = Store(fake, "b", write=True)
    data = b"hello"
    r = s.put("src/a/v1/2023/2023-07-01/" + V + "/x.bin", data, "application/octet-stream")
    assert r["status"] == "written"
    assert calls[0]["IfNoneMatch"] == "*"
    assert calls[0]["ContentMD5"] == base64.b64encode(hashlib.md5(data).digest()).decode()
    assert r["sha256"] == hashlib.sha256(data).hexdigest()


def test_412_identical_counts_as_done_and_different_stops():
    fake = FakeS3()
    s = Store(fake, "b", write=True)
    k = "src/a/v1/2023/2023-07-01/" + V + "/x.bin"
    s.put(k, b"same")
    puts_before = s.puts
    assert s.put(k, b"same")["status"] == "exists-identical"
    assert s.puts == puts_before
    with pytest.raises(OverwriteRefused):
        s.put(k, b"different")
    assert fake.objects[k][0] == b"same"


def test_409_bucket_lock_is_resolved_like_412():
    fake = FakeS3(locked=("src/",))
    s = Store(fake, "b", write=True)
    k = "src/a/v1/2023/2023-07-01/" + V + "/x.bin"
    fake.objects[k] = (b"old", None)
    # a client that does not send If-None-Match would meet the lock's 409
    s2 = Store(fake, "b", write=True)
    orig = fake.put_object
    fake.put_object = lambda **kw: orig(**{**kw, "IfNoneMatch": None})
    with pytest.raises(OverwriteRefused):
        s2.put(k, b"new")
    assert s2.put(k, b"old")["status"] == "exists-identical"
    assert fake.objects[k][0] == b"old"
    del s


def test_dry_run_never_writes():
    fake = FakeS3()
    s = Store(fake, "b", write=False)
    r = s.put("src/a/v1/2023/2023-07-01/" + V + "/x.bin", b"x")
    assert r["status"] == "dry-run" and not fake.objects and not any(c[0] == "put" for c in fake.calls)


def test_from_env_without_credentials(monkeypatch):
    for n in ("S3_ENDPOINT", "S3_BUCKET", "S3_KEY_ID", "S3_SECRET"):
        monkeypatch.delenv(n, raising=False)
    s = Store.from_env(write=False)
    assert not s.online and not s.write
    with pytest.raises(Exception):
        Store.from_env(write=True)


def test_listing_pages():
    fake = FakeS3(page_size=2)
    s = Store(fake, "b", write=True)
    for i in range(5):
        s.put(f"run/r/j/{i:06d}.json", b"{}")
    assert len(list(s.list("run/r/"))) == 5


def test_endpoint_base_drops_bucket_path_and_whitespace():
    from feeder.store import endpoint_base
    assert endpoint_base(" https://abc.r2.cloudflarestorage.com/wl-feed/ \n") == "https://abc.r2.cloudflarestorage.com"
    assert endpoint_base("https://abc.r2.cloudflarestorage.com") == "https://abc.r2.cloudflarestorage.com"


def test_settings_check_reports_shapes_only():
    from feeder.store import settings_check
    out = settings_check({"S3_ENDPOINT": "https://" + "a" * 32 + ".r2.cloudflarestorage.com/wl-feed",
                          "S3_KEY_ID": "b" * 32, "S3_SECRET": "c" * 64, "S3_BUCKET": "x"})
    assert out == ("endpoint is an R2 account host: yes; endpoint had a path: yes; key id is 32 hex: yes; "
                   "secret is 64 hex: yes")
    assert "b" * 8 not in out and "c" * 8 not in out
