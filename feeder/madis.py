"""MADIS mesonet (LDAD) hourly netCDF-3 files -> one lossless Parquet table per UTC day.

Source: the public archive, one gzipped netCDF-3 file per hour (``YYYYMMDD_HH00.gz``, observations HH:00:00 to
HH:59:59 UTC). Every record of every provider is kept, with the configured variables (``madis.columns``) and, for
the configured value fields (``madis.flagged``), their QC flags (``madis.flag_suffixes``, e.g. the DD summary value
and the QC results word) when the file has them.

Lossless rules:
- values are stored exactly as read (no unit conversion, no rounding, no masking): a fill value or a
  ``missing_value`` stays the value it is in the file, bit for bit (floats are compared as their bit patterns);
- a variable the file does not have is stored as null for that hour's rows, so "not in the file" is distinct from
  any fill value;
- character variables keep every byte, padding included;
- each variable's type is fixed per dataset version (``madis.types``); a file with another type fails the day
  instead of being converted;
- each row carries ``_hour`` (its source file) and ``_run``, so each file's record order is restored exactly by
  sorting one hour's rows on (``_run``, dataProvider, stationId, observationTime bits). Rows are stored sorted by
  provider, station and time, which is what makes the table small.
The schema metadata (key ``madis``) lists every source file (name, URL, sha256 and size of the .gz and of the
netCDF, rows, absent variables) and every kept variable's attributes (units, _FillValue, missing_value, ...).
Before upload the stored bytes are read back and compared field by field with the parsed source; a table that
does not match is not uploaded. The .gz files themselves are not kept.

Gap rule: an hour the archive does not have (404) is a source gap (listed as a missing step; the day stays
incomplete, so a later run tries it again); an hour that still fails after the host's retries (5xx) or that cannot
be read fails the day.
"""

import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import zlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import log, par
from .manifest import DatasetEntry, iso
from .noaa import GroupResult, day_steps, fill
from .store import src_key

META_KEY = b"madis"
SCHEMA = "madis-day/1"
ORDER = "_hour, then _run, dataProvider, stationId, observationTime (float64 bits)"
_ARROW = {"c": pa.binary(), "b": pa.int8(), "h": pa.int16(), "i": pa.int32(), "f": pa.float32(), "d": pa.float64()}
_BITS = {"b": np.uint8, "h": np.uint16, "i": np.uint32, "f": np.uint32, "d": np.uint64}
READ_ATTEMPTS = 2  # downloads of one hour whose gzip stream does not decompress


class NotLossless(Exception):
    """The file cannot be stored under this dataset version without losing or converting something."""


# ---------------------------------------------------------------- layout

def flag_names(spec):
    return [f + s for f in spec["flagged"] for s in spec["flag_suffixes"]]


def column_names(spec):
    """Stored variables in table order: the configured columns, then each flagged field's flags."""
    return list(spec["columns"]) + flag_names(spec)


def typecode(spec, name):
    t = spec["types"]
    if name in t:
        return t[name]
    for s in spec["flag_suffixes"]:
        if name.endswith(s) and name[: -len(s)] in spec["flagged"]:
            return t[s]
    raise KeyError(name)


def hour_url(spec, when):
    return fill(spec["url"], when)


def file_name(when):
    return when.strftime("%Y%m%d_%H00") + ".gz"


# ---------------------------------------------------------------- parsing

def _native(a):
    """Native byte order with the same bits (netCDF-3 is big-endian): a byte swap, never a value conversion."""
    a = np.ascontiguousarray(a)
    if a.dtype.isnative:
        return a
    return a.byteswap().view(a.dtype.newbyteorder("="))


def _attr(v):
    if isinstance(v, bytes):
        return v.decode("latin-1")
    a = np.asarray(v)
    if a.dtype.kind == "f":
        vals = [float(x).hex() for x in a.ravel()]
    elif a.dtype.kind in "iu":
        vals = [int(x) for x in a.ravel()]
    else:
        vals = [str(x) for x in a.ravel()]
    return {"dtype": a.dtype.newbyteorder("=").name, "values": vals}


class Hour:
    """One parsed file: ``cols`` {name: native numpy array (n,) or uint8 (n, width) for characters, or None when
    the file does not have the variable}."""

    def __init__(self, index, rows, cols, attrs, absent):
        self.index = index
        self.rows = rows
        self.cols = cols
        self.attrs = attrs
        self.absent = absent


def parse_hour(nc_bytes, spec, index=0):
    """Reads the kept variables of one netCDF-3 file (bytes, not gzipped) exactly as stored in it."""
    from scipy.io import netcdf_file

    nc = netcdf_file(io.BytesIO(nc_bytes), "r", mmap=False, maskandscale=False)
    try:
        variables = nc.variables
        for name in spec["required"]:
            if name not in variables:
                raise NotLossless(f"required variable absent: {name}")
        rows = int(variables[spec["required"][0]].shape[0])
        cols, attrs, absent = {}, {}, []
        for name in column_names(spec):
            if name not in variables:
                cols[name] = None
                absent.append(name)
                continue
            v = variables[name]
            tc, want = v.typecode(), typecode(spec, name)
            if tc != want:
                raise NotLossless(f"type of {name}: {tc}, this version stores {want}")
            data = np.array(v[:], copy=True)
            if data.shape[:1] != (rows,) or (tc != "c" and data.ndim != 1) or data.ndim > 2:
                raise NotLossless(f"shape of {name}: {data.shape}")
            if tc == "c":
                data = np.ascontiguousarray(data).view(np.uint8).reshape(rows, -1)
            else:
                data = _native(data)
            cols[name] = data
            attrs[name] = {k: _attr(x) for k, x in sorted(v._attributes.items())}
    finally:
        nc.close()
    return Hour(index, rows, cols, attrs, absent)


def _svals(u8):
    """uint8 (n, w) -> numpy S{w} with the same bytes (fixed width, so the comparison is exact)."""
    n, w = u8.shape
    if w == 0:
        return np.zeros(n, "S1")
    return np.ascontiguousarray(u8).view(f"S{w}").ravel()


def _key_arrays(spec, cols):
    prov, sid, obs = spec["order_key"]
    return _svals(cols[prov]), _svals(cols[sid]), cols[obs].view(np.uint64)


def order_runs(prov, sid, obsbits):
    """``_run`` for one file: the record order split into maximal runs that strictly increase in the key
    (provider, station, time bits). Within a run the key is unique, so the file order is the sort by
    (run, key)."""
    n = len(obsbits)
    if n == 0:
        return np.zeros(0, np.uint32)
    o = np.lexsort((obsbits, sid, prov))
    sp, ss, so = prov[o], sid[o], obsbits[o]
    new = np.ones(n, bool)
    new[1:] = (sp[1:] != sp[:-1]) | (ss[1:] != ss[:-1]) | (so[1:] != so[:-1])
    rank = np.empty(n, np.int64)
    rank[o] = np.cumsum(new) - 1
    brk = np.ones(n, bool)
    brk[1:] = rank[1:] <= rank[:-1]
    return (np.cumsum(brk) - 1).astype(np.uint32)


def restore_order(prov, sid, obsbits, run):
    """The permutation that puts one file's stored rows back in the file's record order."""
    return np.lexsort((obsbits, sid, prov, run))


# ---------------------------------------------------------------- table

def _binary(u8):
    n, w = u8.shape
    offs = np.arange(n + 1, dtype=np.int32) * np.int32(w)
    return pa.Array.from_buffers(pa.binary(), n, [None, pa.py_buffer(offs), pa.py_buffer(u8.tobytes())])


def _arrow(spec, name, data, rows):
    t = _ARROW[typecode(spec, name)]
    if data is None:
        return pa.nulls(rows, t)
    if t == pa.binary():
        return _binary(data)
    return pa.array(data, type=t)


def build_table(spec, hours, meta):
    """The day table (sorted for storage) from parsed hours; ``meta`` goes in the schema metadata."""
    names = column_names(spec)
    parts = {n: [] for n in ["_hour", "_run"] + names}
    for h in hours:
        run = order_runs(*_key_arrays(spec, h.cols))
        parts["_hour"].append(pa.array(np.full(h.rows, h.index, np.uint8)))
        parts["_run"].append(pa.array(run))
        for n in names:
            parts[n].append(_arrow(spec, n, h.cols[n], h.rows))
    types = {"_hour": pa.uint8(), "_run": pa.uint32(), **{n: _ARROW[typecode(spec, n)] for n in names}}
    arrays = [pa.concat_arrays(parts[n]) if parts[n] else pa.array([], types[n]) for n in parts]
    t = pa.table(arrays, names=list(parts))
    prov, sid, obs = spec["order_key"]
    t = t.sort_by([(prov, "ascending"), (sid, "ascending"), (obs, "ascending"), ("_hour", "ascending"),
                   ("_run", "ascending")])
    return t.replace_schema_metadata({META_KEY: json.dumps(meta, sort_keys=True, separators=(",", ":"),
                                                           ensure_ascii=True).encode("ascii")})


def to_parquet(spec, table):
    """zstd Parquet, one row group; dictionary pages for every column but ``_run`` (measured smallest)."""
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", compression_level=int(spec.get("zstd_level", 19)),
                   use_dictionary=[c for c in table.column_names if c != "_run"],
                   row_group_size=max(1, table.num_rows), write_statistics=True)
    return buf.getvalue()


def _u8_of(arr):
    """A binary Arrow array -> uint8 (n, w) when every value has the same width, else None."""
    arr = arr.combine_chunks() if isinstance(arr, pa.ChunkedArray) else arr
    n = len(arr)
    if n == 0:
        return np.zeros((0, 0), np.uint8)
    _, obuf, dbuf = arr.buffers()
    offs = np.frombuffer(obuf, np.int32, count=n + 1, offset=arr.offset * 4)
    lens = np.diff(offs)
    w = int(lens[0])
    if not np.all(lens == w):
        return None
    if w == 0:
        return np.zeros((n, 0), np.uint8)
    data = np.frombuffer(dbuf, np.uint8, count=n * w, offset=int(offs[0]))
    return data.reshape(n, w)


def decode_back_ok(spec, blob, hours):
    """Reads the stored bytes back and compares every field of every row with the parsed source, in each file's
    record order. Returns (ok, reason)."""
    t = pq.read_table(io.BytesIO(blob))
    names = column_names(spec)
    if t.column_names != ["_hour", "_run"] + names:
        return False, "columns"
    if t.num_rows != sum(h.rows for h in hours):
        return False, "rows"
    hour_col = t.column("_hour").to_numpy()
    seen = 0
    prov, sid, obs = spec["order_key"]
    for h in hours:
        idx = np.flatnonzero(hour_col == h.index)
        if len(idx) != h.rows:
            return False, f"rows of hour {h.index}"
        seen += len(idx)
        sub = t.take(pa.array(idx))
        kp, ks = _u8_of(sub.column(prov)), _u8_of(sub.column(sid))
        if kp is None or ks is None or sub.column(obs).null_count:
            return False, "key columns"
        ob = sub.column(obs).combine_chunks().to_numpy(zero_copy_only=True).view(np.uint64)
        run = sub.column("_run").to_numpy()
        sub = sub.take(pa.array(restore_order(_svals(kp), _svals(ks), ob, run)))
        for n in names:
            col = sub.column(n).combine_chunks()
            src = h.cols[n]
            if src is None:
                if col.null_count != h.rows:
                    return False, f"{n}: absent in the file but stored"
                continue
            if col.null_count:
                return False, f"{n}: nulls"
            tc = typecode(spec, n)
            if tc == "c":
                got = _u8_of(col)
                if got is None or not np.array_equal(got, src):
                    return False, f"{n}: bytes differ"
            else:
                got = col.to_numpy(zero_copy_only=True)
                if got.dtype != src.dtype or not np.array_equal(got.view(_BITS[tc]), src.view(_BITS[tc])):
                    return False, f"{n}: bits differ"
    if seen != t.num_rows:
        return False, "rows outside the source hours"
    return True, None


# ---------------------------------------------------------------- one day

def _fetch(http, spec, when):
    """(Response, netCDF bytes or None, read error type or None) for one hour."""
    url = hour_url(spec, when)
    err = None
    for _ in range(READ_ATTEMPTS):
        r = http.fetch(url)
        if not r.ok:
            return r, None, None
        try:
            return r, gzip.decompress(r.body), None
        except (OSError, EOFError, zlib.error) as e:
            err = type(e).__name__
            log("madis-read", file_name(when), err)
    return r, None, err


def process_day(ctx, spec, day, supersedes=None):
    """Fetches the day's hours, builds and checks the day table, uploads it. Returns a noaa.GroupResult."""
    res = GroupResult()
    entry = DatasetEntry(spec["id"], spec["ver"], ctx.vintage)
    entry["supersedes"] = supersedes
    res.entries.append(entry)
    steps = day_steps(day, int(spec.get("cadence_min", 60)))
    entry["steps"]["expected"] = len(steps)
    known = set(spec.get("known_gaps", []))
    hours, files, problems = [], [], []
    ahead = max(1, int(spec.get("fetch_ahead", 2)))
    with cf.ThreadPoolExecutor(max_workers=ahead) as pool:
        got = par.prefetch(pool, lambda w: (w, _fetch(ctx.http, spec, w)), steps, ahead)
        for i, (when, (r, raw, read_err)) in enumerate(got):
            ctx.breaker.observe_http(ctx.http)
            valid = iso(when)
            entry.add_source(**r.source_record(valid_utc=valid))
            if raw is None:
                entry.add_missing(valid, r.status, r.tries, known_gap=valid in known)
                if read_err:
                    problems.append(f"read: {read_err}")
                continue
            try:
                h = parse_hour(raw, spec, index=when.hour)
            except Exception as err:  # format change or damaged file: never stored half-read
                entry.add_missing(valid, r.status, r.tries, known_gap=False)
                problems.append(f"parse: {type(err).__name__}")
                log("madis-parse", file_name(when), type(err).__name__)
                continue
            entry["steps"]["got"] += 1
            hours.append(h)
            files.append({"name": file_name(when), "url": r.url, "hour": when.hour, "sha256": r.sha256,
                          "bytes": len(r.body), "nc_sha256": hashlib.sha256(raw).hexdigest(), "nc_bytes": len(raw),
                          "rows": h.rows, "absent": h.absent, "attrs": h.attrs})
            del raw
    rows = sum(h.rows for h in hours)
    if hours:
        meta = {"schema": SCHEMA, "dataset": spec["id"], "ver": spec["ver"], "day": day.isoformat(), "order": ORDER,
                "columns": column_names(spec), "files": files,
                "missing_hours": [m["valid_utc"] for m in entry["steps"]["missing"]]}
        table = build_table(spec, hours, meta)
        blob = to_parquet(spec, table)
        del table
        ok, why = decode_back_ok(spec, blob, hours)
        entry["checks"]["decode_back"] = "pass" if ok else "fail"
        entry["checks"]["rows"] = rows
        if ok:
            key = src_key(spec["id"], spec["ver"], day, ctx.vintage, f"{spec['id']}-{day.strftime('%Y%m%d')}.parquet")
            pr = ctx.store.put(key, blob, "application/vnd.apache.parquet")
            entry.add_object(pr, covers={"hours": len(hours), "rows": rows})
            res.written.append((pr, blob))
            res.bytes += pr["bytes"]
        else:
            res.failed = True
            res.reasons.append(f"decode-back failed: {why}")
            log("madis-decode-back", day.isoformat(), why)
    entry.finish_status()
    if entry["checks"].get("decode_back") == "fail":
        entry["status"] = "partial"
    missing = entry["steps"]["missing"]
    hard = sorted({m["http"] for m in missing if m["http"] not in (0, 404)})
    if problems or hard:
        res.failed = True
        res.reasons.append("fetch: " + "; ".join(sorted(set(problems)) + [f"http {s}" for s in hard if s != 200]))
    if ctx.breaker.gap_failed(missing):
        res.failed = True
        res.reasons.append(f"gaps: {spec['id']}")
    log("madis", day.isoformat(), f"hours={len(hours)}", f"missing={len(missing)}", f"rows={rows}",
        f"bytes={res.bytes}")
    return res
