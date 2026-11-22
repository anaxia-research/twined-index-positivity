# Exact R-chamber Fourier-Jacobi coefficients of 1/Phi_g for g = 1A, 2A, 3A, 4B (twined, unorbifolded).
import sys, os, pickle
from math import isqrt
from ser import Ctx
HERE = os.path.dirname(os.path.abspath(__file__))
SHAPE = {1: {1: 24}, 2: {1: 8, 2: 8}, 3: {1: 6, 3: 6}, 4: {1: 4, 2: 2, 4: 4}}
_G = {}
def gen(N, Dmax):
    from genera import build
    key = (N, Dmax)
    if key not in _G:
        fn = os.path.join(HERE, 'gen_N%d_%d.pkl' % key)
        _G[key] = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else build(N, Dmax)
    return _G[key]
def psi_coeffs(M, NQ, N, Y=None, Dmax=None, twine=1):
    """coefficients c_{m,g^twine}(n,l) of psi_{m,g^twine} in the R-chamber; N = order of g (g = 1A,2A,3A,4B for N=1..4)."""
    from math import gcd
    need = 4*(NQ+1)*(M+1) + 4
    if Dmax is None: Dmax = max(200, need)
    if N == 1:
        G = gen(2, Dmax); cg = {0: G['cg'][0]}; Nn = 1
    else:
        G = gen(N, Dmax); cg = G['cg']; Nn = N
    ordr = Nn//gcd(twine, Nn) if Nn > 1 else 1
    shape = SHAPE[ordr]
    cfun = lambda t: {D: int(v) for D, v in cg[(t*twine) % Nn].items()}
    QD = G['QD']
    if Y is None: Y = 2*isqrt((NQ+2)*(M+2)) + M + NQ + 6
    C = Ctx(NQ+1, Y); NQ1 = NQ+1
    cache = {}
    def ct(t):
        if (t*twine) % Nn not in cache: cache[(t*twine) % Nn] = cfun(t)
        return cache[(t*twine) % Nn]
    LW = {}
    for L in range(1, M+2):
        A = C.zero()
        for t in range(1, L+1):
            if L % t: continue
            c = ct(t); Lt = L//t
            for k in range(0, NQ1//t + 1):
                jm = isqrt(4*k*Lt + 1)
                for j in range(-jm, jm+1):
                    D = 4*k*Lt - j*j
                    if D < -1: continue
                    v = c.get(QD*D, 0)
                    if v and abs(j*t) <= Y and k*t <= NQ1:
                        A[k*t, j*t + Y] += Lt*v
        LW[L] = A
    E = {0: C.mono(0, 0)}
    for L in range(1, M+2):
        S = C.zero()
        for k in range(1, L+1): S = S + C.mul(LW[k], E[L-k])
        assert all(int(x) % L == 0 for x in S.flat)
        E[L] = S//L
    # B = 1/[prod (1-q^n)^{m1-4} prod_{M>1}(1-q^{Mn})^{mM} (1-q^n y)^2 (1-q^n/y)^2]
    c = [0]*(NQ1+1); c[0] = 1
    ex = dict(shape); ex[1] = ex.get(1, 0) - 4
    for Mm, e in ex.items():
        for k in range(1, NQ1//Mm + 1):
            st = Mm*k
            if e > 0:
                for _ in range(e):
                    for i in range(st, NQ1+1): c[i] += c[i-st]
            else:
                for _ in range(-e):
                    for i in range(NQ1, st-1, -1): c[i] -= c[i-st]
    B = C.zero()
    for n in range(NQ1+1): B[n, Y] = c[n]
    for k in range(1, NQ1+1):
        f = C.zero(); g = C.zero()
        for i in range(0, NQ1//k + 1):
            if i <= Y: f[k*i, i+Y] = i+1; g[k*i, -i+Y] = i+1
        B = C.mul(B, f); B = C.mul(B, g)
    res = {}
    for m in range(1, M+1):
        Gm = C.mul(E[m+1], B)
        for n in range(-1, NQ+1):
            row = Gm[n+1]
            for l in range(0, 2*m + 1):
                s = 0
                for j in range(l+1, Y+1):
                    v = row[j+Y]
                    if v: s += (j-l)*int(v)
                res[(m, n, l)] = s
    return res
def dstar(res, m, n, l): return (-1)**(l+1)*res[(m, n, l)]
if __name__ == '__main__':
    M, NQ = 3, 3
    out = {}
    for N in (1, 2, 3, 4):
        out[N] = psi_coeffs(M, NQ, N)
    out['4sq'] = psi_coeffs(M, NQ, 4, twine=2)
    for N in (1, 2, 3, 4, '4sq'):
        r = out[N]
        rows = []
        for (P2, Q2) in [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3)]:
            m, n = P2, Q2   # m = P^2/2 -> FJ index; n = Q^2/2
            rows.append(((2*n, 2*m), [dstar(r, m, n, l) if l <= 2*m else None for l in (0, 1, 2, 3, 4)]))
        print(N, rows)
    pickle.dump(out, open(os.path.join(HERE, 'small_M3.pkl'), 'wb'))
