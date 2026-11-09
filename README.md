# sa-lab-11-5

Tools that fetch public NOAA and IEM weather data, compact it, checksum it and upload it to private storage. Workflows run on manual dispatch only.

- Gridded fields are stored exactly: the source's packed integers in compressed 128x128 tiles, or the original message where that is smaller. Each object is decoded back and compared bit for bit before the day is accepted, and the day manifest records the packing factors and checksums of every step.
- 8-bit display tiles are an optional output, off by default; they can be rebuilt from the exact grids.
- `python -m feeder.selftest` runs offline checks; `IN_DAY=YYYY-MM-DD python -m feeder.dryplan` sizes one day from index files and listings, with one sampled step per grid (no uploads).
