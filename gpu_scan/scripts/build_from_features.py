"""Rebuild the primitive solid from the features marked in the GPU Feature Marker page.
Features apply in `order`: add = union, cut = subtract. The feature labelled "Heatsink" also gets its fin slots.
Usage: python build_from_features.py features_dir_or_json params.json out_prefix"""
import json, sys, os, glob
from build123d import Box, Cylinder, Pos, Align, export_step, export_stl

src, P, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
if os.path.isdir(src):
    feats = [json.load(open(p)) for p in glob.glob(os.path.join(src, '*.json'))]
    feats = [f.get('data', f) for f in feats]
else:
    feats = json.load(open(src))
feats.sort(key=lambda f: f['order'])
C = (Align.MIN, Align.MIN, Align.MIN)

def box(x0, x1, y0, y1, z0, z1):
    return Pos(min(x0, x1), min(y0, y1), z0) * Box(abs(x1 - x0), abs(y1 - y0), z1 - z0, align=C)

def solid_of(f):
    if f['kind'] == 'cyl':
        return Pos(f['cx'], f['cy'], f['z0']) * Cylinder(f['r'], f['z1'] - f['z0'], align=(Align.CENTER, Align.CENTER, Align.MIN))
    return box(f['x0'], f['x1'], f['y0'], f['y1'], f['z0'], f['z1'])

def fin_slots(f):
    hs = P['heatsink']; p, n, t = hs['fin_pitch'], hs['fin_count'], hs['fin_thickness']
    y0, y1 = sorted((f['y0'], f['y1'])); x0, x1 = sorted((f['x0'], f['x1']))
    span = (n - 1) * p + t; f0 = (y0 + y1) / 2 - span / 2
    zr, zt = hs['z_fin_root'], f['z1'] + 1
    cuts = [box(x0 - 1, x1 + 1, f0 + k * p + t, f0 + (k + 1) * p, zr, zt) for k in range(n - 1)]
    cuts += [box(x0 - 1, x1 + 1, y0 - 1, f0, zr, zt), box(x0 - 1, x1 + 1, f0 + span, y1 + 1, zr, zt)]
    return cuts

part, log = None, []
for f in feats:
    s = solid_of(f)
    if f['op'] == 'add':
        part = s if part is None else part + s
        if f.get('label', '').strip().lower() == 'heatsink':
            for c in fin_slots(f):
                part -= c
    else:
        if part is None:
            continue
        before = part.volume; part -= s
        log.append((f.get('label', ''), before - part.volume))
part = part.clean()
print('valid', part.is_valid, 'solids', len(part.solids()), 'volume %.0f' % part.volume)
for lab, dv in log:
    if dv < 0.05:
        print('  cut removed nothing:', lab)
export_step(part, out + '.step')
export_stl(part, out + '.stl', tolerance=0.02, angular_tolerance=0.2)
json.dump([{k: v for k, v in f.items()} for f in feats], open(out + '_features.json', 'w'), indent=1)
