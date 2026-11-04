import os
import tempfile
import unittest

import numpy as np

from noisemsg import DecodeError, Params, decode_text, demodulate, encode_text, modulate, turbo


def awgn(x, snr_db, rng):
    """Add white Gaussian noise so that signal power / noise power = snr_db."""
    sig = np.mean(x ** 2)
    return x + rng.normal(0, np.sqrt(sig / 10 ** (snr_db / 10)), len(x))


class TurboTests(unittest.TestCase):
    def test_roundtrip_noiseless(self):
        rng = np.random.default_rng(1)
        u = rng.integers(0, 2, turbo.K).astype(np.uint8)
        c = turbo.encode(u)
        self.assertEqual(len(c), turbo.N_CODED)
        bits, _ = turbo.decode((c * 2.0 - 1) * 8)
        self.assertTrue(np.array_equal(bits, u))

    def test_corrects_errors(self):
        rng = np.random.default_rng(2)
        u = rng.integers(0, 2, turbo.K).astype(np.uint8)
        # Es/N0 = -3 dB per coded bit (Eb/N0 = 1.8 dB): raw hard BER ~ 16%
        var = 0.5 / 10 ** -0.3
        llr = (turbo.encode(u) * 2.0 - 1) + rng.normal(0, np.sqrt(var), turbo.N_CODED)
        bits, _ = turbo.decode(llr * 2 / var)
        self.assertTrue(np.array_equal(bits, u))


class ModemTests(unittest.TestCase):
    MSG = "The quick brown fox — jumps over 13 lazy dögs. ✓"

    def test_wav_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            f = os.path.join(d, "n.wav")
            encode_text(self.MSG, f, Params(key="k"))
            self.assertEqual(decode_text(f, Params(key="k")), self.MSG)

    def test_looks_like_white_noise(self):
        x = modulate(self.MSG.encode(), Params(key="k"))
        spec = np.abs(np.fft.rfft(x[: 2 ** 15]))[:-1] ** 2
        bands = spec.reshape(32, -1).mean(axis=1)
        self.assertLess(bands.max() / bands.min(), 1.5)           # flat spectrum
        self.assertLess(abs(np.corrcoef(x[:-1], x[1:])[0, 1]), 0.02)  # no structure

    def test_heavy_noise(self):
        rng = np.random.default_rng(3)
        p = Params(key="k")
        x = awgn(modulate(self.MSG.encode(), p), -12.0, rng)      # noise 16x signal power
        self.assertEqual(demodulate(x, p).decode(), self.MSG)

    def test_unknown_offset_inverted_and_embedded(self):
        rng = np.random.default_rng(4)
        p = Params(key="k")
        x = -modulate(self.MSG.encode(), p)
        x = np.concatenate([rng.normal(0, 0.3, 12345), x, rng.normal(0, 0.3, 999)])
        self.assertEqual(demodulate(awgn(x, -8, rng), p).decode(), self.MSG)

    def test_multi_block(self):
        rng = np.random.default_rng(5)
        msg = bytes(rng.integers(0, 256, 300, dtype=np.uint8))
        p = Params(key="k", chips_per_bit=8)
        self.assertEqual(demodulate(awgn(modulate(msg, p), -8, rng), p), msg)

    def test_wrong_key_fails(self):
        x = modulate(self.MSG.encode(), Params(key="right"))
        with self.assertRaises(DecodeError):
            demodulate(x, Params(key="wrong"))

    def test_pure_noise_fails(self):
        x = np.random.default_rng(6).normal(0, 0.3, 100000)
        with self.assertRaises(DecodeError):
            demodulate(x, Params(key="k"))

    def test_empty_and_sps(self):
        p = Params(key="k", sps=2, chips_per_bit=8)
        self.assertEqual(demodulate(modulate(b"", p), p), b"")
        self.assertEqual(demodulate(modulate(b"hi", p), p), b"hi")


if __name__ == "__main__":
    unittest.main()
