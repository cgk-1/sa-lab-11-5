"""Box flags on a fixed 2-degree lattice, from gridded inputs and warning polygons only.

A box-day is flagged if any of:
  (1) the day's maximum gust is >= 58 mph at >= 8 grid cells inside the box plus a 0.25-degree buffer;
  (2) the day's maximum MESH is >= 25.4 mm at >= 25 grid cells inside the buffered box;
  (3) >= 2 severe-thunderstorm / tornado warning polygons touch the buffered box.
The lattice starts at 125 W / 24 N and covers 125 W - 65 W, 24 N - 50 N. Only box ids are stored, no counts.
"""

import io
import math
import struct
import zipfile

import numpy as np

LON0, LAT0, LON1, LAT1 = -125.0, 24.0, -65.0, 50.0
SIZE = 2.0
BUFFER = 0.25


def all_boxes():
    nx = int(round((LON1 - LON0) / SIZE))
    ny = int(round((LAT1 - LAT0) / SIZE))
    return [(i, j) for j in range(ny) for i in range(nx)]


def box_id(i, j):
    lon = LON0 + i * SIZE
    lat = LAT0 + j * SIZE
    return f"W{int(round(-lon)):03d}N{int(round(lat)):02d}"


def box_bounds(i, j, buffer=BUFFER):
    lon = LON0 + i * SIZE
    lat = LAT0 + j * SIZE
    return lon - buffer, lat - buffer, lon + SIZE + buffer, lat + SIZE + buffer


def boxes_for_point(lat, lon):
    """Lattice boxes whose buffered extent contains the point (a point near an edge belongs to 2-4 boxes)."""
    out = []
    i0 = math.floor((lon - LON0 - BUFFER) / SIZE)
    i1 = math.floor((lon - LON0 + BUFFER) / SIZE)
    j0 = math.floor((lat - LAT0 - BUFFER) / SIZE)
    j1 = math.floor((lat - LAT0 + BUFFER) / SIZE)
    nx = int(round((LON1 - LON0) / SIZE))
    ny = int(round((LAT1 - LAT0) / SIZE))
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            if 0 <= i < nx and 0 <= j < ny:
                x0, y0, x1, y1 = box_bounds(i, j)
                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    out.append((i, j))
    return out


def grid_rule(day_max, lats, lons, threshold, min_cells):
    """Box ids where ``day_max >= threshold`` at ``>= min_cells`` cells (inside the buffered box)."""
    v = np.asarray(day_max).ravel()
    with np.errstate(invalid="ignore"):
        hit = np.nonzero(np.isfinite(v) & (v >= threshold))[0]
    la = np.asarray(lats).ravel()[hit]
    lo = np.asarray(lons).ravel()[hit]
    counts = {}
    for a, b in zip(la.tolist(), lo.tolist()):
        for box in boxes_for_point(a, b):
            counts[box] = counts.get(box, 0) + 1
    return sorted(box_id(*b) for b, n in counts.items() if n >= min_cells)


# ---------------------------------------------------------------- polygons

def _seg_intersect(p1, p2, q1, q2):
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and \
            min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return (o1 == 0 and on_seg(p1, p2, q1)) or (o2 == 0 and on_seg(p1, p2, q2)) or \
        (o3 == 0 and on_seg(q1, q2, p1)) or (o4 == 0 and on_seg(q1, q2, p2))


def _point_in_ring(pt, ring):
    x, y = pt
    inside = False
    n = len(ring)
    for k in range(n):
        x1, y1 = ring[k]
        x2, y2 = ring[(k + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xi:
                inside = not inside
    return inside


def polygon_touches_rect(rings, rect):
    """``rings``: list of [(lon, lat), ...]; ``rect``: (x0, y0, x1, y1)."""
    x0, y0, x1, y1 = rect
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    for ring in rings:
        if not ring:
            continue
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        if max(xs) < x0 or min(xs) > x1 or max(ys) < y0 or min(ys) > y1:
            continue
        if any(x0 <= p[0] <= x1 and y0 <= p[1] <= y1 for p in ring):
            return True
        if any(_point_in_ring(c, ring) for c in corners):
            return True
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            if any(_seg_intersect(a, b, e0, e1) for e0, e1 in edges):
                return True
    return False


def polygon_rule(polygons, min_polygons=2):
    """Box ids touched by ``>= min_polygons`` polygons (each polygon = list of rings)."""
    counts = {}
    for rings in polygons:
        pts = [p for r in rings for p in r]
        if not pts:
            continue
        bx0 = min(p[0] for p in pts)
        bx1 = max(p[0] for p in pts)
        by0 = min(p[1] for p in pts)
        by1 = max(p[1] for p in pts)
        for (i, j) in all_boxes():
            r = box_bounds(i, j)
            if bx1 < r[0] or bx0 > r[2] or by1 < r[1] or by0 > r[3]:
                continue
            if polygon_touches_rect(rings, r):
                counts[(i, j)] = counts.get((i, j), 0) + 1
    return sorted(box_id(*b) for b, n in counts.items() if n >= min_polygons)


def read_shp_polygons(shp):
    """Polygons (shape type 5) from the bytes of a .shp file: list of polygons, each a list of rings."""
    if len(shp) < 100 or struct.unpack(">i", shp[0:4])[0] != 9994:
        raise ValueError("not a shapefile")
    out = []
    pos = 100
    while pos + 8 <= len(shp):
        _, clen = struct.unpack(">ii", shp[pos:pos + 8])
        rec = shp[pos + 8:pos + 8 + clen * 2]
        pos += 8 + clen * 2
        if len(rec) < 4:
            break
        (stype,) = struct.unpack("<i", rec[0:4])
        if stype == 0:
            continue
        if stype not in (5, 15, 25):
            raise ValueError(f"shape type {stype} is not a polygon")
        nparts, npoints = struct.unpack("<ii", rec[36:44])
        parts = list(struct.unpack(f"<{nparts}i", rec[44:44 + 4 * nparts]))
        p0 = 44 + 4 * nparts
        pts = struct.unpack(f"<{2 * npoints}d", rec[p0:p0 + 16 * npoints])
        xy = list(zip(pts[0::2], pts[1::2]))
        rings = [xy[parts[k]:(parts[k + 1] if k + 1 < nparts else npoints)] for k in range(nparts)]
        out.append(rings)
    return out


def polygons_from_zip(zip_bytes):
    polys = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.lower().endswith(".shp"):
                polys.extend(read_shp_polygons(zf.read(name)))
    return polys
