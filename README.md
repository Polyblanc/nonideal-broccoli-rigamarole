# nonideal-broccoli-rigamarole
We all make mistakes sometimes, this one included

## noisemsg — text hidden in a noise-like WAV, protected by a turbo code

Encodes text into a WAV file that sounds (and measures) like white noise, and
decodes it again even when the recording is buried in additional noise.

```
pip install -r requirements.txt
python -m noisemsg encode "meet at dawn" -o msg.wav --key hunter2
python -m noisemsg decode msg.wav --key hunter2
python -m unittest discover -s tests
```

**How it works**

1. `[len:2][utf-8 text][crc32:4]`, padded to 1024-bit blocks, XOR-whitened with a key-seeded PRNG.
2. Each block gets a rate-1/3 LTE-style turbo code: two 8-state RSC encoders (13,15 octal), QPP interleaver, trellis termination. Decoded with 16-iteration log-MAP BCJR using soft inputs (`noisemsg/turbo.py`).
3. BPSK symbols are spread with a key-seeded ±1 PN chip sequence (16 chips/bit by default), so the waveform is a flat-spectrum pseudo-random stream.
4. A keyed PN preamble, found by FFT cross-correlation, gives timing, polarity, amplitude and noise-variance estimates (which scale the LLRs). The signal can sit anywhere in a longer recording and be sign-inverted.

**Measured** (default settings, additive white Gaussian noise, 44-byte message, 6 trials per level):
decodes 6/6 down to −13 dB signal-to-noise (noise 20× the power of the signal); fails at −14 dB. Net rate ≈ 115 bytes/s.
`--spread` trades speed for robustness (shared by both ends, like `--key`, `--sps`, `--rate`).

**Limits:** this is obscurity, not encryption — the key only seeds the PN sequences. There is no clock-drift or
frequency-offset tracking, so lossy codecs (MP3), resampling or over-the-air playback are not handled; the
decoder expects the 16-bit mono/stereo WAV at the same sample rate. Max message 65535 bytes.
