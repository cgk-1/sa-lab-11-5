"""Lossless compaction.

Parquet (rt): a delimited text reply becomes a Parquet table whose columns hold every field exactly as served
(text, never converted), plus a ``_row`` column with the row's position in the reply. The header, the line ending,
the final line ending, any leading comment lines and the text encoding go in the schema metadata. Re-serialising
the table (or the union of its day slices, ordered by ``_row``) gives the reply's bytes; the sha256 is checked on
every file. A reply that cannot be represented this way is kept as served, zstd-compressed.

Also: zstd for JSON and text, and deterministic tar bundles of as-served files.
"""

import hashlib
import io
import json
import tarfile

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import zstandard

RT_KEY = b"rt"
SLOW_PATH_MAX = 64 * 1024 * 1024


class NotRoundTrippable(Exception):
    pass


# ---------------------------------------------------------------- zstd / tar

def zstd(data, level=19):
    return zstandard.ZstdCompressor(level=level, write_checksum=True).compress(data)


def unzstd(data):
    return zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 34)


def tar_bytes(members):
    """``members``: list of (name, bytes). Deterministic: fixed mtime / owner / mode, given order."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def untar(data):
    out = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tf:
        for m in tf.getmembers():
            out.append((m.name, tf.extractfile(m).read()))
    return out


def member_records(members):
    return [{"name": n, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in members]


# ---------------------------------------------------------------- Parquet (rt)

def _decode(data):
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def _eol(text):
    crlf = text.count("\r\n")
    lf = text.count("\n")
    cr = text.count("\r")
    if crlf and crlf == lf == cr:
        return "\r\n"
    if cr == 0:
        return "\n"
    raise NotRoundTrippable("mixed or bare CR line endings")


def _split_quote_aware(line, delim):
    out, start, inq = [], 0, False
    for i, ch in enumerate(line):
        if ch == '"':
            inq = not inq
        elif ch == delim and not inq:
            out.append(line[start:i])
            start = i + 1
    if inq:
        raise NotRoundTrippable("unbalanced quotes")
    out.append(line[start:])
    return out


def _records_quote_aware(body, eol):
    """Splits on eol outside quotes."""
    recs, start, inq, i, n, le = [], 0, False, 0, len(body), len(eol)
    while i < n:
        ch = body[i]
        if ch == '"':
            inq = not inq
        elif not inq and body.startswith(eol, i):
            recs.append(body[start:i])
            i += le
            start = i
            continue
        i += 1
    if inq:
        raise NotRoundTrippable("unbalanced quotes")
    recs.append(body[start:])
    return recs


def _unique_names(header):
    names, seen = [], {}
    for i, h in enumerate(header):
        base = (h.strip().strip('"') or f"col{i}").replace("\x00", "")
        if base == "_row":
            base = "_row_"
        k = seen.get(base, 0)
        seen[base] = k + 1
        names.append(base if k == 0 else f"{base}.{k}")
    return names


def csv_to_table(data, delim=","):
    """Bytes of a delimited reply -> pyarrow Table (all string columns + _row) with rt metadata."""
    text, enc = _decode(data)
    if text == "":
        raise NotRoundTrippable("empty reply")
    eol = _eol(text)
    final_eol = text.endswith(eol)
    body = text[:-len(eol)] if final_eol else text
    # leading comment lines
    preamble = []
    while body.startswith("#"):
        cut = body.find(eol)
        if cut < 0:
            preamble.append(body)
            body = ""
            break
        preamble.append(body[:cut])
        body = body[cut + len(eol):]
    if body == "":
        raise NotRoundTrippable("no header")
    head_end = body.find(eol)
    header_line = body if head_end < 0 else body[:head_end]
    rows_text = "" if head_end < 0 else body[head_end + len(eol):]
    header = _split_quote_aware(header_line, delim)
    names = _unique_names(header)
    ncol = len(header)
    meta = {"delim": delim, "eol": eol, "final_eol": final_eol, "encoding": enc, "header": header,
            "preamble": preamble, "has_rows": head_end >= 0, "source_sha256": hashlib.sha256(data).hexdigest()}
    if head_end < 0:
        cols = {n: pa.array([], pa.string()) for n in names}
    elif '"' not in rows_text and "\r" not in rows_text.replace(eol, "") and enc == "utf-8":
        # fast path: no quoting at all
        opts = pacsv.ParseOptions(delimiter=delim, quote_char=False, escape_char=False, double_quote=False,
                                  newlines_in_values=False, ignore_empty_lines=False)
        ropts = pacsv.ReadOptions(column_names=names, autogenerate_column_names=False, block_size=1 << 26)
        copts = pacsv.ConvertOptions(column_types={n: pa.string() for n in names}, null_values=[],
                                     strings_can_be_null=False, quoted_strings_can_be_null=False, check_utf8=True)
        try:
            t = pacsv.read_csv(io.BytesIO(rows_text.encode("utf-8") + eol.encode("utf-8")), read_options=ropts,
                               parse_options=opts, convert_options=copts)
        except (pa.ArrowInvalid, pa.ArrowTypeError) as err:
            raise NotRoundTrippable(f"row shape: {err}") from err
        cols = {n: t.column(n) for n in names}
    else:
        if len(rows_text) > SLOW_PATH_MAX:
            raise NotRoundTrippable("quoted reply too large for the exact slow path")
        recs = _records_quote_aware(rows_text, eol)
        fields = [_split_quote_aware(r, delim) for r in recs]
        if any(len(f) != ncol for f in fields):
            raise NotRoundTrippable("row with a different field count")
        cols = {n: pa.array([f[i] for f in fields], pa.string()) for i, n in enumerate(names)}
    n = len(next(iter(cols.values()))) if cols else 0
    arrays = [pa.array(range(n), pa.uint32())] + [cols[nm] for nm in names]
    t = pa.table(arrays, names=["_row"] + names)
    return t.replace_schema_metadata({RT_KEY: json.dumps(meta, ensure_ascii=False).encode("utf-8")})


def table_to_bytes(table):
    """Re-serialises a (possibly re-assembled) rt table into the reply's bytes."""
    meta = json.loads(table.schema.metadata[RT_KEY].decode("utf-8"))
    delim, eol = meta["delim"], meta["eol"]
    enc = meta["encoding"]
    table = table.sort_by("_row") if table.num_rows > 1 else table
    parts = [p + eol for p in meta["preamble"]]
    head = delim.join(meta["header"])
    out = io.BytesIO()
    out.write("".join(parts).encode(enc))
    out.write(head.encode(enc))
    data_cols = [c for c in table.column_names if c != "_row"]
    if meta["has_rows"]:
        out.write(eol.encode(enc))
        n = table.num_rows
        step = 200_000
        for s in range(0, n, step):
            sl = table.slice(s, step)
            joined = pc.binary_join_element_wise(*[sl.column(c) for c in data_cols], delim) if len(data_cols) > 1 \
                else sl.column(data_cols[0])
            text = eol.join(joined.to_pylist())
            out.write(text.encode(enc))
            if s + step < n:
                out.write(eol.encode(enc))
    if meta["final_eol"]:
        out.write(eol.encode(enc))
    return out.getvalue()


def table_to_parquet(table):
    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd", compression_level=9)
    return buf.getvalue()


def parquet_to_table(data):
    return pq.read_table(io.BytesIO(data))


def roundtrip_ok(parquet_parts, source_sha256):
    """True if the union of the Parquet parts re-serialises to bytes with ``source_sha256``."""
    tables = [parquet_to_table(p) for p in parquet_parts]
    t = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    t = t.replace_schema_metadata(tables[0].schema.metadata)
    return hashlib.sha256(table_to_bytes(t)).hexdigest() == source_sha256


def slice_by_day(table, time_columns):
    """{YYYY-MM-DD: sub-table} using the first 10 characters of the first matching time column."""
    names = table.column_names
    col = next((c for c in time_columns if c in names), None)
    if col is None:
        raise NotRoundTrippable("time column not found")
    if table.num_rows == 0:
        return {}
    days = pc.utf8_slice_codeunits(table.column(col), 0, 10)
    out = {}
    for d in pc.unique(days).to_pylist():
        out[d] = table.filter(pc.equal(days, d))
    return out
