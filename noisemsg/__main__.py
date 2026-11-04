import argparse
import sys

from .modem import DecodeError, Params, decode_text, encode_text


def main(argv=None):
    ap = argparse.ArgumentParser(prog="noisemsg", description="Hide text in turbo-coded noise (WAV).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--key", default="", help="shared secret seeding the PN sequences")
        s.add_argument("--spread", type=int, default=16, help="chips per coded bit (default 16)")
        s.add_argument("--sps", type=int, default=1, help="samples per chip (default 1)")
        s.add_argument("--rate", type=int, default=44100, help="sample rate")

    e = sub.add_parser("encode", help="text -> noise WAV")
    e.add_argument("text", nargs="?", help="message (default: read stdin)")
    e.add_argument("-o", "--output", required=True)
    common(e)
    d = sub.add_parser("decode", help="noise WAV -> text")
    d.add_argument("input")
    common(d)

    a = ap.parse_args(argv)
    p = Params(key=a.key, chips_per_bit=a.spread, sps=a.sps, sample_rate=a.rate)
    try:
        if a.cmd == "encode":
            encode_text(a.text if a.text is not None else sys.stdin.read(), a.output, p)
        else:
            info = {}
            print(decode_text(a.input, p, info=info))
            print(f"[chip SNR {info['chip_snr_db']:.1f} dB, turbo iterations {info['iterations']}]", file=sys.stderr)
    except (DecodeError, ValueError) as ex:
        sys.exit(f"error: {ex}")


if __name__ == "__main__":
    main()
