"""Rebuild the primitive solid from the features marked in the GPU Feature Marker page.

Features apply in `order`: add = union, cut = subtract. The feature labelled "Heatsink" also gets its fin slots.
Overlap rule: where component-level add shapes (z0 >= 4.9, not the heatsink) overlap, the overlap is its own
feature. Its top is fitted to the scan (median) inside the overlap, and each shape's own top is refitted
from the scan over the part of it that nobody else covers.
Usage: python build_from_features.py features_dir_or_json params.json hmap_mm.npz out_prefix"""
import json, sys, os, glob, itertools
import numpy as np
from scipy import ndimage as ndi
from build123d import Box, Cylinder, Pos, Align, export_step, export_stl

src, P, hmp, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3], sys.argv[4]
if os.path.isdir(src):
    feats = []
    for p in sorted(glob.glob(os.path.join(src, '*.json'))):
        f = json.load(open(p)); feats.append(dict(f.get('data', f), id=f.get('id') or os.path.basename(p)[:-5]))
else:
    feats = json.load(open(src))
    feats = feats.get('features', feats) if isinstance(feats, dict) else feats
    for i, f in enumerate(feats):
        f.setdefault('id', 'f%03d' % i)
feats.sort(key=lambda f: f['order'])
C = (Align.MIN, Align.MIN, Align.MIN)
MIN_OVERLAP = 1.0   # mm^2; smaller overlaps are drawing slop, not features

def box(x0, x1, y0, y1, z0, z1):
    return Pos(min(x0, x1), min(y0, y1), z0) * Box(abs(x1 - x0), abs(y1 - y0), z1 - z0, align=C)

def prism(f, z0, z1):
    if f['kind'] == 'cyl':
        return Pos(f['cx'], f['cy'], z0) * Cylinder(f['r'], z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return box(f['x0'], f['x1'], f['y0'], f['y1'], z0, z1)

def fin_slots(f):
    hs = P['heatsink']; p, n, t = hs['fin_pitch'], hs['fin_count'], hs['fin_thickness']
    y0, y1 = sorted((f['y0'], f['y1'])); x0, x1 = sorted((f['x0'], f['x1']))
    span = (n - 1) * p + t; f0 = (y0 + y1) / 2 - span / 2
    zr, zt = hs['z_fin_root'], f['z1'] + 1
    cuts = [box(x0 - 1, x1 + 1, f0 + k * p + t, f0 + (k + 1) * p, zr, zt) for k in range(n - 1)]
    cuts += [box(x0 - 1, x1 + 1, y0 - 1, f0, zr, zt), box(x0 - 1, x1 + 1, f0 + span, y1 + 1, zr, zt)]
    return cuts

# ---- scan raster for fitting ----
hm = np.load(hmp); H = ndi.median_filter(np.nan_to_num(hm['hmap'], nan=0), 5); mn = hm['mn']; res = float(hm['res'])
ys, xs = mn[1] + (np.arange(H.shape[0]) + .5) * res, mn[0] + (np.arange(H.shape[1]) + .5) * res
def mask(f):
    if f['kind'] == 'cyl':
        b = (f['cx'] - f['r'], f['cx'] + f['r'], f['cy'] - f['r'], f['cy'] + f['r'])
    else:
        b = (min(f['x0'], f['x1']), max(f['x0'], f['x1']), min(f['y0'], f['y1']), max(f['y0'], f['y1']))
    c0, c1 = np.searchsorted(xs, b[0]), np.searchsorted(xs, b[1]); r0, r1 = np.searchsorted(ys, b[2]), np.searchsorted(ys, b[3])
    m = np.zeros(H.shape, bool); m[r0:r1, c0:c1] = True
    if f['kind'] == 'cyl':
        X, Y = np.meshgrid(xs[c0:c1], ys[r0:r1]); m[r0:r1, c0:c1] = (X - f['cx']) ** 2 + (Y - f['cy']) ** 2 <= f['r'] ** 2
    return m

comp = [f for f in feats if f['op'] == 'add' and f['z0'] >= 4.9 and f.get('label', '').strip().lower() != 'heatsink']
M = {f['id']: mask(f) for f in comp}
cnt = sum(M.values()) if M else None
cell = res * res
report = []

# exclusive refit for every component shape that overlaps something
overl = {f['id'] for f, g in itertools.combinations(comp, 2) if (M[f['id']] & M[g['id']]).sum() * cell >= MIN_OVERLAP for f in (f, g)}
top = {}
for f in comp:
    top[f['id']] = f['z1']
    if f['id'] in overl:
        ex = M[f['id']] & (cnt == 1)
        if ex.sum() * cell >= MIN_OVERLAP:
            top[f['id']] = round(float(np.median(H[ex])), 1)
        report.append((f.get('label', ''), 'own area', f['z1'], top[f['id']]))

# overlap groups: cells covered by >= 2 shapes, keyed by exactly which shapes cover them
ids = [f['id'] for f in comp]; byid = {f['id']: f for f in comp}
key = np.zeros(H.shape, dtype=object)
groups = {}
if M:
    stack = np.stack([M[i] for i in ids])
    multi = stack.sum(0) >= 2
    for r, c in zip(*np.where(multi)):
        k = tuple(i for i, v in zip(ids, stack[:, r, c]) if v)
        groups.setdefault(k, []).append((r, c))
ov_feats = []
for k, cells in groups.items():
    if len(cells) * cell < MIN_OVERLAP:
        continue
    rr, cc = np.array(cells).T
    h = round(float(np.median(H[rr, cc])), 1)
    z0 = max(byid[i]['z0'] for i in k)
    ov_feats.append((k, z0, h))
    report.append((' ∩ '.join(byid[i].get('label', '') for i in k), 'overlap', None, h))

def component_block():
    solid = None
    for f in comp:
        s = prism(f, f['z0'], max(top[f['id']], f['z0'] + .1))
        others = [g for g in comp if g['id'] != f['id'] and g['id'] in overl and f['id'] in overl and (M[f['id']] & M[g['id']]).sum() * cell >= MIN_OVERLAP]
        for g in others:
            s -= prism(g, f['z0'] - 1, 60)
        solid = s if solid is None else solid + s
    for k, z0, h in ov_feats:
        s = None
        for i in k:
            p_ = prism(byid[i], z0, max(h, z0 + .1)); s = p_ if s is None else s & p_
        for g in comp:      # trim parts belonging to a larger overlap set
            if g['id'] not in k and (sum(M[i] for i in k) == len(k))[M[g['id']]].any():
                s -= prism(g, z0 - 1, 60)
        if s is not None and s.volume > 1e-3:
            solid = solid + s
    return solid

part, log, done_block = None, [], False
for f in feats:
    if f in comp:
        if done_block:
            continue
        s = component_block(); done_block = True
    else:
        s = prism(f, f['z0'], f['z1'])
    if f['op'] == 'add':
        part = s if part is None else part + s
        if f.get('label', '').strip().lower() == 'heatsink':
            for c in fin_slots(f):
                part -= c
    elif part is not None:
        before = part.volume; part -= s
        log.append((f.get('label', ''), before - part.volume))
part = part.clean()
print('valid', part.is_valid, 'solids', len(part.solids()), 'volume %.0f' % part.volume)
for lab, dv in log:
    if dv < 0.05:
        print('  cut removed nothing:', lab)
for r in report:
    print('  refit %-40s %-8s %s -> %.1f' % (r[0], r[1], '' if r[2] is None else r[2], r[3]))
export_step(part, out + '.step')
export_stl(part, out + '.stl', tolerance=0.02, angular_tolerance=0.2)
json.dump(dict(features=feats, refits=[dict(what=r[0], region=r[1], drawn_top=r[2], fitted_top=r[3]) for r in report]), open(out + '_features.json', 'w'), indent=1)
