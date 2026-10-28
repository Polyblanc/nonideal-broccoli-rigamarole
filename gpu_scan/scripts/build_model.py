"""Primitive-only solid model (cuboids add/subtract, cylinders, circular fillet fillers) -> STEP + STL.
Usage: python build_model.py params.json blocks.json out_prefix"""
import json, sys, time
from build123d import Box, Cylinder, Pos, Align, Compound, export_step, export_stl

P = json.load(open(sys.argv[1])); blocks = json.load(open(sys.argv[2])); out = sys.argv[3]
C = (Align.MIN, Align.MIN, Align.MIN)

def box(x0, x1, y0, y1, z0, z1):
    return Pos(x0, y0, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=C)

def cyl(x, y, r, z0, z1):
    return Pos(x, y, z0) * Cylinder(r, z1 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))

t = time.time()
pc = P['pcb']; z0, z1 = pc['z0'], pc['z1']
pcb = box(*pc['main'], z0, z1) + box(*pc['tab'], z0, z1)
for c in pc['cuts']:
    pcb -= box(*c, z0 - 1, z1 + 1)
pcb -= box(*pc['key_notch'], z0 - 1, z1 + 1)

ue = P['underside_envelope']
under = box(*ue['rect'], ue['z0'], ue['z1'])

hs = P['heatsink']; x0, x1, y0, y1 = hs['rect']
sink = box(x0, x1, y0, y1, hs['z_base0'], hs['z_top'])
# fins run along x; slots between fins are subtractive cuboids down to the fin root
p, n, tf = hs['fin_pitch'], hs['fin_count'], hs['fin_thickness']
span = (n - 1) * p + tf; f0 = (y0 + y1) / 2 - span / 2
for k in range(n - 1):
    sy0 = f0 + k * p + tf
    sink -= box(x0 - 1, x1 + 1, sy0, sy0 + p - tf, hs['z_fin_root'], hs['z_top'] + 1)
# outer margins beyond the fin array
sink -= box(x0 - 1, x1 + 1, y0 - 1, f0, hs['z_fin_root'], hs['z_top'] + 1)
sink -= box(x0 - 1, x1 + 1, f0 + span, y1 + 1, hs['z_fin_root'], hs['z_top'] + 1)
# corner notch, with a circular filler (square minus cylinder) in its re-entrant corner
nx0, nx1, ny0, ny1 = hs['notch']; r = hs['notch_fillet_r']
sink -= box(nx0 - 1, nx1, ny0, ny1 + 1, hs['z_base0'] - 1, hs['z_top'] + 1)
filler = box(nx1 - r, nx1, ny0, ny0 + r, hs['z_base0'], hs['z_top']) - cyl(nx1 - r, ny0 + r, r, hs['z_base0'] - 1, hs['z_top'] + 1)
sink += filler
# the filler is solid to the top; re-cut the slots through it so fins stay consistent
for k in range(n - 1):
    sy0 = f0 + k * p + tf
    if sy0 < ny0 + r:
        sink -= box(nx1 - r - 1, nx1 + 0.01, sy0, sy0 + p - tf, hs['z_fin_root'], hs['z_top'] + 1)

cp = P['capacitors']
caps = [cyl(x, y, cp['r'], cp['z0'], cp['z1']) for x, y in cp['xy']]
blk = [box(*b) for b in blocks]
print('primitives built %.1fs' % (time.time() - t))

parts = {'pcb': pcb, 'underside_envelope': under, 'heatsink': sink}
comp = Compound(children=[pcb, under, sink] + caps + blk)
solid = pcb.fuse(under, sink, *caps, *blk).clean()
print('fused %.1fs, volume %.0f mm^3, solids %d' % (time.time() - t, solid.volume, len(solid.solids())))
export_step(solid, out + '.step')
export_stl(solid, out + '.stl', tolerance=0.02, angular_tolerance=0.2)
# also a multi-body STEP keeping named groups separate (handy for editing in CAD)
for name, s in parts.items(): s.label = name
grp = Compound(children=[pcb, under, sink, Compound(children=caps, label='capacitors'), Compound(children=blk, label='component_envelope_blocks')], label='gpu_primitives')
export_step(grp, out + '_bodies.step')
print('done %.1fs' % (time.time() - t))
