"""Rate-1/3 turbo code (LTE-style) with an iterative log-MAP decoder.

Two identical 8-state recursive systematic convolutional (RSC) encoders,
feedback polynomial 1+D^2+D^3 (octal 13), parity polynomial 1+D+D^3 (octal 15),
joined by a quadratic permutation polynomial (QPP) interleaver.  Each
constituent encoder is trellis-terminated (3 tail steps), as in LTE.

Bit convention: LLR = ln P(bit=1) / P(bit=0); positive means "1".
"""
from __future__ import annotations

import numpy as np

K = 1024            # information bits per block
MEM = 3             # encoder memory
NS = 1 << MEM       # trellis states
TAIL = MEM
N_CODED = 3 * K + 4 * TAIL   # coded bits per block
_QPP_F1, _QPP_F2 = 31, 64    # LTE parameters for K = 1024

# ---------------------------------------------------------------- trellis
# state = (s1, s2, s3) packed as s1<<2 | s2<<1 | s3
_NEXT = np.zeros((NS, 2), dtype=np.int64)
_PAR = np.zeros((NS, 2), dtype=np.int64)
for _s in range(NS):
    _s1, _s2, _s3 = (_s >> 2) & 1, (_s >> 1) & 1, _s & 1
    for _u in (0, 1):
        _w = _u ^ _s2 ^ _s3
        _PAR[_s, _u] = _w ^ _s1 ^ _s3
        _NEXT[_s, _u] = (_w << 2) | (_s1 << 1) | _s2

# incoming branches per next-state, for the alpha recursion
_INC_S = np.zeros((NS, 2), dtype=np.int64)
_INC_U = np.zeros((NS, 2), dtype=np.int64)
_cnt = [0] * NS
for _s in range(NS):
    for _u in (0, 1):
        _n = _NEXT[_s, _u]
        _INC_S[_n, _cnt[_n]] = _s
        _INC_U[_n, _cnt[_n]] = _u
        _cnt[_n] += 1
assert all(c == 2 for c in _cnt)

_SGN_U = np.array([-1.0, 1.0])
_SGN_P = 2.0 * _PAR - 1.0
_NEG = -1e30

_i = np.arange(K, dtype=np.int64)
PERM = (_QPP_F1 * _i + _QPP_F2 * _i * _i) % K
assert len(set(PERM.tolist())) == K, "QPP interleaver is not a permutation"


def _rsc_encode(u: np.ndarray):
    """Return (parity[K], tail_sys[3], tail_par[3]) for info bits u."""
    s = 0
    par = np.empty(len(u), dtype=np.uint8)
    for k, b in enumerate(u):
        par[k] = _PAR[s, b]
        s = _NEXT[s, b]
    tsys = np.empty(TAIL, dtype=np.uint8)
    tpar = np.empty(TAIL, dtype=np.uint8)
    for k in range(TAIL):
        b = ((s >> 1) ^ s) & 1          # s2 ^ s3 drives feedback to zero
        tsys[k] = b
        tpar[k] = _PAR[s, b]
        s = _NEXT[s, b]
    assert s == 0
    return par, tsys, tpar


def encode(bits: np.ndarray) -> np.ndarray:
    """Encode K info bits -> N_CODED coded bits.

    Layout: [systematic K][parity1 K][parity2 K][tail: x1(3) p1(3) x2(3) p2(3)]
    """
    u = np.asarray(bits, dtype=np.uint8)
    assert u.shape == (K,)
    p1, x1, q1 = _rsc_encode(u)
    p2, x2, q2 = _rsc_encode(u[PERM])
    return np.concatenate([u, p1, p2, x1, q1, x2, q2]).astype(np.uint8)


def _bcjr(ls: np.ndarray, lp: np.ndarray, la: np.ndarray) -> np.ndarray:
    """Log-MAP BCJR on a terminated trellis; returns a-posteriori LLRs."""
    T = len(ls)
    lsa = ls + la
    gamma = 0.5 * (_SGN_U[None, None, :] * lsa[:, None, None]
                   + _SGN_P[None, :, :] * lp[:, None, None])      # (T, NS, 2)

    alpha = np.full((T + 1, NS), _NEG)
    alpha[0, 0] = 0.0
    for t in range(T):
        c = alpha[t][_INC_S] + gamma[t][_INC_S, _INC_U]
        a = np.logaddexp(c[:, 0], c[:, 1])
        alpha[t + 1] = a - a.max()

    beta = np.full((T + 1, NS), _NEG)
    beta[T, 0] = 0.0
    for t in range(T - 1, -1, -1):
        d = gamma[t] + beta[t + 1][_NEXT]
        b = np.logaddexp(d[:, 0], d[:, 1])
        beta[t] = b - b.max()

    m = alpha[:-1, :, None] + gamma + beta[1:][:, _NEXT]          # (T, NS, 2)
    l0 = np.logaddexp.reduce(m[:, :, 0], axis=1)
    l1 = np.logaddexp.reduce(m[:, :, 1], axis=1)
    return l1 - l0


def decode(llr: np.ndarray, max_iter: int = 16):
    """Decode N_CODED channel LLRs -> (K hard bits, iterations used)."""
    llr = np.asarray(llr, dtype=np.float64)
    assert llr.shape == (N_CODED,)
    ls, lp1, lp2 = llr[:K], llr[K:2 * K], llr[2 * K:3 * K]
    t = llr[3 * K:]
    x1, q1, x2, q2 = t[:3], t[3:6], t[6:9], t[9:12]

    sys1 = np.concatenate([ls, x1])
    par1 = np.concatenate([lp1, q1])
    sys2 = np.concatenate([ls[PERM], x2])
    par2 = np.concatenate([lp2, q2])

    la1 = np.zeros(K + TAIL)
    prev = None
    hard = (ls > 0).astype(np.uint8)
    it = 0
    for it in range(1, max_iter + 1):
        l1 = _bcjr(sys1, par1, la1)
        le1 = (l1 - sys1 - la1)[:K]

        la2 = np.concatenate([le1[PERM], np.zeros(TAIL)])
        l2 = _bcjr(sys2, par2, la2)
        le2 = (l2 - sys2 - la2)[:K]

        la1 = np.zeros(K + TAIL)
        la1[PERM] = le2                                   # de-interleave
        total = ls + le1 + la1[:K]
        hard = (total > 0).astype(np.uint8)
        if prev is not None and np.array_equal(hard, prev):
            break
        prev = hard
    return hard, it
