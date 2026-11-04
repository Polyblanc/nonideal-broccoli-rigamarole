"""Noise-like acoustic modem: turbo-coded, direct-sequence spread spectrum.

Pipeline (encode):  text -> [len|data|crc32] -> 1024-bit blocks -> whiten
    -> turbo encode -> BPSK -> multiply by key-seeded +/-1 PN chips
    -> PN preamble + payload -> WAV.
The waveform is a pseudo-random +/-1 chip stream, i.e. white noise.  Without
the key it is statistically indistinguishable from noise (this is obscurity,
NOT encryption).
"""
from __future__ import annotations

import hashlib
import struct
import wave
import zlib
from dataclasses import dataclass

import numpy as np

from . import turbo

BLOCK_BYTES = turbo.K // 8
HEADER_BYTES = 2        # uint16 payload length
CRC_BYTES = 4
MAX_PAYLOAD = 0xFFFF


class DecodeError(Exception):
    pass


@dataclass(frozen=True)
class Params:
    key: str = ""
    chips_per_bit: int = 16     # spreading gain: higher = more noise-robust, slower
    sps: int = 1                # samples per chip (>1 narrows the bandwidth)
    sample_rate: int = 44100
    amplitude: float = 0.5      # peak fraction of full scale
    preamble_chips: int = 2048
    pad_seconds: float = 0.25   # noise lead-in / lead-out

    @property
    def block_samples(self) -> int:
        return turbo.N_CODED * self.chips_per_bit * self.sps

    @property
    def bitrate(self) -> float:
        """Net information bits per second (before framing overhead)."""
        return self.sample_rate / (self.sps * self.chips_per_bit) * turbo.K / turbo.N_CODED


def _rng(p: Params, label: str, idx: int = 0) -> np.random.Generator:
    h = hashlib.sha256(f"{p.key}|{label}|{idx}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "big"))


def _pn(p: Params, label: str, n: int, idx: int = 0) -> np.ndarray:
    return _rng(p, label, idx).integers(0, 2, n, dtype=np.int8) * 2.0 - 1.0


def _preamble(p: Params) -> np.ndarray:
    return np.repeat(_pn(p, "preamble", p.preamble_chips), p.sps)


# ------------------------------------------------------------------ framing
def _frame(data: bytes) -> bytes:
    if len(data) > MAX_PAYLOAD:
        raise ValueError(f"message too long ({len(data)} > {MAX_PAYLOAD} bytes)")
    body = struct.pack(">H", len(data)) + data
    body += struct.pack(">I", zlib.crc32(body))
    pad = -len(body) % BLOCK_BYTES
    return body + bytes(pad)


def _whiten(p: Params, bits: np.ndarray, blk: int) -> np.ndarray:
    return bits ^ _rng(p, "whiten", blk).integers(0, 2, turbo.K, dtype=np.uint8)


# --------------------------------------------------------------- modulation
def modulate(data: bytes, p: Params = Params()) -> np.ndarray:
    """Return float samples in [-amplitude, amplitude]."""
    frame = _frame(data)
    parts = [_pn(p, "pad_head", int(p.pad_seconds * p.sample_rate))]
    parts.append(_preamble(p))
    for blk in range(len(frame) // BLOCK_BYTES):
        chunk = np.frombuffer(frame[blk * BLOCK_BYTES:(blk + 1) * BLOCK_BYTES], np.uint8)
        bits = _whiten(p, np.unpackbits(chunk), blk)
        sym = turbo.encode(bits).astype(np.float64) * 2 - 1
        chips = np.repeat(sym, p.chips_per_bit) * _pn(p, "chips", turbo.N_CODED * p.chips_per_bit, blk)
        parts.append(np.repeat(chips, p.sps))
    parts.append(_pn(p, "pad_tail", int(p.pad_seconds * p.sample_rate)))
    return np.concatenate(parts) * p.amplitude


# ------------------------------------------------------------ demodulation
def _find_sync(x: np.ndarray, p: Params):
    """FFT cross-correlation against the preamble. -> (start, amp, noise_var)."""
    tpl = _preamble(p)
    n = 1 << int(np.ceil(np.log2(len(x) + len(tpl))))
    corr = np.fft.irfft(np.fft.rfft(x, n) * np.conj(np.fft.rfft(tpl, n)), n)
    corr = corr[: len(x) - len(tpl) + 1]
    if len(corr) <= 0:
        raise DecodeError("signal shorter than the preamble")
    pos = int(np.argmax(np.abs(corr)))
    peak = corr[pos]
    sigma = np.std(corr)
    if abs(peak) < 6.0 * sigma:
        raise DecodeError("no preamble found (wrong key, or no signal)")
    amp = peak / float(np.dot(tpl, tpl))            # signed amplitude estimate
    seg = x[pos:pos + len(tpl)]
    noise_var = float(np.mean((seg - amp * tpl) ** 2))
    return pos, amp, max(noise_var, 1e-12)


def _soft_bits(seg: np.ndarray, p: Params, blk: int, amp: float, nv: float) -> np.ndarray:
    chips = np.repeat(_pn(p, "chips", turbo.N_CODED * p.chips_per_bit, blk), p.sps)
    y = (seg * chips).reshape(turbo.N_CODED, -1).sum(axis=1)
    return 2.0 * amp * y / nv


def demodulate(x: np.ndarray, p: Params = Params(), max_iter: int = 16, info: dict | None = None) -> bytes:
    x = np.asarray(x, dtype=np.float64)
    pos, amp, nv = _find_sync(x, p)
    # Fold polarity into the amplitude: LLR sign follows amp, so use |amp|
    # for scaling and flip the signal if the channel inverted it.
    if amp < 0:
        x = -x
        amp = -amp
    cur = pos + p.preamble_chips * p.sps
    out = bytearray()
    nblocks = None
    blk = 0
    iters = []
    while nblocks is None or blk < nblocks:
        seg = x[cur:cur + p.block_samples]
        if len(seg) < p.block_samples:
            raise DecodeError(f"signal truncated in block {blk}")
        llr = _soft_bits(seg, p, blk, amp, nv)
        bits, it = turbo.decode(llr, max_iter)
        iters.append(it)
        out += np.packbits(_whiten(p, bits, blk)).tobytes()
        if blk == 0:
            (n,) = struct.unpack(">H", out[:HEADER_BYTES])
            nblocks = -(-(HEADER_BYTES + n + CRC_BYTES) // BLOCK_BYTES)
        cur += p.block_samples
        blk += 1
    (n,) = struct.unpack(">H", out[:HEADER_BYTES])
    end = HEADER_BYTES + n
    payload = bytes(out[HEADER_BYTES:end])
    (crc,) = struct.unpack(">I", out[end:end + CRC_BYTES])
    if info is not None:
        info.update(sync=pos, amplitude=amp, noise_var=nv, iterations=iters,
                    chip_snr_db=10 * np.log10(amp * amp / nv))
    if crc != zlib.crc32(bytes(out[:end])):
        raise DecodeError("CRC mismatch: message too corrupted to recover")
    return payload


# --------------------------------------------------------------------- WAV
def write_wav(path: str, x: np.ndarray, sample_rate: int) -> None:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def read_wav(path: str):
    with wave.open(path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise DecodeError("only 16-bit PCM WAV files are supported")
    x = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
    return x.reshape(-1, ch).mean(axis=1), rate


# ------------------------------------------------------------- convenience
def encode_text(text: str, path: str, p: Params = Params()) -> None:
    write_wav(path, modulate(text.encode("utf-8"), p), p.sample_rate)


def decode_text(path: str, p: Params = Params(), **kw) -> str:
    x, rate = read_wav(path)
    if rate != p.sample_rate:
        raise DecodeError(f"sample rate {rate} != expected {p.sample_rate}")
    return demodulate(x, p, **kw).decode("utf-8")
