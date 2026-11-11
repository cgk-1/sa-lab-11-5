# sa-lab-11-5

Tools that fetch public NOAA and IEM weather data, compact it, checksum it and upload it to private storage. The backfill workflow runs on manual dispatch only; the daily append also runs on a schedule (08:23 UTC, with a 17:13 UTC retry).

- Gridded fields are stored exactly: the source's packed integers in compressed 128x128 tiles, or the original message where that is smaller. Each object is decoded back and compared bit for bit before the day is accepted, and the day manifest records the packing factors and checksums of every step.
- 8-bit display tiles are an optional output, off by default; they can be rebuilt from the exact grids.
- `python -m feeder.selftest` runs offline checks; `IN_DAY=YYYY-MM-DD python -m feeder.dryplan` sizes one day from index files and listings, with one sampled step per grid (no uploads).
- The gridded lane decodes, encodes and checks steps in one worker process per CPU (`FEEDER_PROCS`, blank = all CPUs, 1 = in-process), downloads ahead of the encoder, and logs its peak memory per group (`peak-mem`). The stored bytes are the same as with one process.
- The mesonet lane (`madis`, from 2021-10-01) keeps every report of the hourly MADIS LDAD mesonet files as one zstd Parquet table per UTC day: the configured variables and their QC flags exactly as stored in the file (no unit conversion; fill values stay values; a variable a file lacks is null), each file's record order recoverable, the source names and sha256 in the table and the manifest. The table is read back and compared field by field before upload; the hourly files are not kept. An hour the archive lacks (404) is a gap and is tried again by later runs.
