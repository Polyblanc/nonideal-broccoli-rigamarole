# Scan sources (paper trail)

The raw scan archives are not stored in this repository. This file records what was received, what each output was made from, and what was deleted.

## v2: `NewerGPU.zip`. Current source, received 2026-10-30

- Upload: `21de3703-NewerGPU.zip`, 13,024,386 bytes
- sha256 `d9ee9f3fb23553dff122783313803a14e420f6b669bc369b4673fbd01668565d`
- Notes from the user: about 5× smaller and more refined than v1, but it has large holes caused by reflections.

| File | Bytes | sha256 |
|---|---|---|
| NewerGPU.obj | 9,871,634 | `44862961694d1cc480e6060b8c1ecd43d0fb065a9ee5fc1c590e2859e3ff37de` |
| NewerGPU.mtl | 204 | `f305228d887449ea65d1a5e399125944ec6815f5a455d421d7bafd7b590f48ad` |
| tex_u1_v1_diffuse.jpg | 5,016,302 | `af9333955b928c00add5041a8a039446bf4e126e9835279f02c49b413a193e7d` |
| tex_u1_v1_normal.jpg | 4,945,392 | `1b1c53f685ea70b56ca91e7796aad68d5224de529b759420d0d5988d05774186` |

- Mesh: 50,180 vertices, 99,973 faces, 59,047 UVs. Exported by Apple ModelIO from RealityCapture 2.0.0.118105.
- Texture: 8192×8192.

## v1: `GPU.zip`. Superseded, archive deleted 2026-10-30

- Upload: `68cf03cc-GPU.zip`, 24,590,584 bytes, received 2026-10-28
- sha256 `ff3d182ee7fdeeccf6a534bdde002dbcecc540487a78c81d04cf498730587e76`
- Deleted at the user's request to save space and replaced by v2. Its extracted files were deleted from the working session too.

| File | Bytes | sha256 |
|---|---|---|
| GPU.obj | 50,523,842 | `872512164402204e9a7441ff08a1c389491d6e5138058fe709fc6fb7af505dbb` |
| GPU.mtl | 204 | `f305228d887449ea65d1a5e399125944ec6815f5a455d421d7bafd7b590f48ad` |
| tex_u1_v1_diffuse.jpg | 4,391,662 | `571801a3fc88126911fea192054958816a685433704ce80c01dc6b035325fa12` |
| tex_u1_v1_normal.jpg | 5,005,810 | `adb3e7bd12e043a7bf1dab0feac951339ae36846f30fa47d43cf09580cb71e7b` |

- Mesh: 240,621 vertices, 477,126 faces, 273,231 UVs, 8192×8192 texture.
- The ChArUco frame and scale derived from it (checker corrected to 19.296 mm, so the print was about 96.5%) are documented in `README.md`.

**Everything currently in `output/`, and the feature list in the GPU Feature Marker page, was made from v1.**

To check whether a file you hold is one of these, run `sha256sum <file>` and compare.
