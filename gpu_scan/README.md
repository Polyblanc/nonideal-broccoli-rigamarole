# GPU scan → to-scale CAD

Source: RealityScan export `GPU.obj` (240k vertices, 477k faces, 8K diffuse + normal texture) of a
low-profile PCIe graphics card lying on a ChArUco board (8×12, DICT_4X4, IDs from 0) printed on A4
and glued to 3 mm acrylic.

## Outputs (`output/`)

| File | What it is |
|---|---|
| `gpu_primitives.step` | **Main deliverable.** A single valid B-rep solid built only from cuboids (added and cut), cylinders, and one circular fillet filler. Units are mm. |
| `gpu_primitives_bodies.step` | The same model as separate bodies (PCB, underside envelope, heatsink, capacitors, component blocks), for editing in CAD. |
| `gpu_primitives.stl` | Mesh of the primitive solid. |
| `gpu_shrinkwrap.stl` / `.3mf` / `.ply` | Watertight, de-noised best-fit mesh of the part, with the board removed (200k faces). |
| `deviation_map.png` | Height of the scan, height of the primitive model, and the difference between them. |
| `preview.png`, `topview_board_frame.png` | Preview renders, and a top view in the final frame (the board text reads correctly, so the model is not mirrored). |

Parasolid (`.x_t`) needs Siemens' licensed kernel. To get one, import the STEP into Onshape (or
any other Parasolid-based CAD) and export it as Parasolid.

**Coordinate frame:** origin at ChArUco corner (0,0); x runs along the card (bracket and display
end at low x); y runs across it (PCIe fingers at low y); z points up from the paper surface.

## Scale: the print was not 100%

- The ChArUco fit (50 corners, 28 markers) is internally consistent: 0.39 mm RMS residual and 0.13% anisotropy.
- At a 20 mm checker size, however, the sheet would measure 218 × 307 mm. A real A4 sheet is 210 × 297 mm.
- Using the paper edges instead, the checker comes out at **19.30 mm**. The printer most likely scaled the page to about 96.5% ("fit to page").
- Two independent checks against the card itself agree with 19.30 mm:
  - The PCB length is **167.7 mm**. The PCIe low-profile spec is 167.65 mm.
  - The fingers from the end to the key notch measure **11.5 mm**. The spec is 11.65 mm.
- The model uses 19.296 mm. If you measure the printed squares and they differ, everything scales linearly: change `checker` in `tomm.py` and re-run.

## Accuracy against the scan (top-surface height, mm)

| Region | Median | Mean abs. error | Within 1 mm | Within 2 mm |
|---|---|---|---|---|
| Everything except the fin field | −0.13 | 0.88 | 82% | 90% |
| Same, with the 1 mm outline band also excluded | −0.13 | 0.75 | 86% | 93% |
| Heatsink fin tips | −0.01 | 0.36 | 98% | 99% |

## Main dimensions (mm)

- **PCB:** x 32.9–200.6, y −101.9 to −48.0, z 3.4–5.0. The card rests on something about 3.4 mm tall that the scan can't see.
- **Finger tab:** x 78.9–178.2, down to y −115.1. Key notch is 1.9 wide and 7.7 deep at x 90.4. There is a retention-hook slot at the far end of the fingers.
- **Heatsink:** 57.5 × 50.5 mm footprint, top at z 32.7, fin root at z 13.2. 33 fins at 1.552 mm pitch, running along x. One corner has a 15.1 × 9.1 notch with an R7 fillet filler.
- **Capacitors:** four cylinders, Ø5.6, at x 51.3, y −61.1 / −70.1 / −79.4 / −88.3, tops at z 12.4.
- **Everything else on the component side:** about 300 stacked 1 mm cuboid slabs, a best-fit envelope from the scan.

## Known limits (please check with calipers)

1. **Fin thickness (0.5 mm) is a guess.** The film over the card bridges the fins, so only the pitch, count and root height could be measured (from the texture and a side view).
2. **The clear film over the card** adds roughly 0.2–0.5 mm everywhere and softens edges. Component blocks are an envelope, not individual parts.
3. **The underside was not scanned.** `underside_envelope` (z 0–3.4) is a placeholder for whatever the card rests on. Delete it if you only want the top side.
4. The STL of the primitive solid has 5 edges where blocks meet only along a line. The STEP is a valid closed solid.

## Rebuild

All dimensions are in `scripts/params.json`; edit them and run:

```sh
pip install build123d opencv-contrib-python-headless trimesh scipy fast-simplification lxml
python scripts/build_model.py scripts/params.json scripts/blocks.json output/gpu_primitives
```

Full pipeline order:

1. `load.py` → `plane.py` → `ortho.py`: find the board plane and render the top view.
2. `scale.py` (ChArUco) → `paper.py` (A4 check) → `tomm.py`: board frame in mm.
3. `ortho.py`: height map in the board frame.
4. `autoblocks.py` → `build_model.py` → `shrinkwrap.py`.
