# Cusp data for Gamma_0(N), N = 2, 3, 4 (g = 2A, 3A, 4B) and the coefficient tables c*(T) used in the tail bounds.
import os, pickle, numpy as np
from math import sqrt, pi, log, exp, gcd
from fractions import Fraction as Fr
import cmath
HERE = os.path.dirname(os.path.abspath(__file__))
DEX = 20000
# eta_g|gamma = const * prod eta(r tau + b)^m ; listed as (r, m); 'lc' = log|const|; w = sum r m / 24
CUSPS = {
 2: {'I': dict(c=0, dl=1, Nx=1, Nc=1, cmin=2, eta=[(1, 8), (2, 8)], lc=0.0),
     'S': dict(c=1, dl=0, Nx=2, Nc=2, cmin=1, eta=[(1, 8), (0.5, 8)], lc=-4*log(2))},
 3: {'I': dict(c=0, dl=1, Nx=1, Nc=1, cmin=3, eta=[(1, 6), (3, 6)], lc=0.0),
     'S': dict(c=1, dl=0, Nx=3, Nc=3, cmin=1, eta=[(1, 6), (1/3, 6)], lc=-3*log(3))},
 4: {'I': dict(c=0, dl=1, Nx=1, Nc=1, cmin=4, eta=[(1, 4), (2, 2), (4, 4)], lc=0.0),
     'S': dict(c=1, dl=0, Nx=4, Nc=4, cmin=1, eta=[(1, 4), (0.5, 2), (0.25, 4)], lc=-5*log(2)),
     'H': dict(c=2, dl=1, Nx=2, Nc=2, cmin=2, eta=[(1, 4), (2, 2), (1, 4)], lc=-2*log(2))},
}
WEIGHT = {2: 8, 3: 6, 4: 5}            # weight of eta_g
for N, cs in CUSPS.items():
    for name, d in cs.items():
        d['A'] = 1/(4*d['Nc']**2); d['w'] = sum(r*m for r, m in d['eta'])/24; d['name'] = name
        assert d['Nc'] == N//gcd(d['c'], N)
def kw(N): return (WEIGHT[N] - 2)/2      # |psi(sigma)| = (t/s)^{kw} ...
# ---- coefficient bound for D > DEX (Lemma: coefficient bounds) ----
# base forms (A, const term h0, K_h = 24 * sum |alpha_i|) of phi_{(r,s)} = A phi01 + h phi_{-2,1}
BASE = {2: [(2, 0, 0), (Fr(2, 3), Fr(4, 3), Fr(4, 3)*24*(2 + 1)), (Fr(2, 3), Fr(2, 3), Fr(2, 3)*24*(2 + 1))],
        3: [(2, 0, 0), (Fr(1, 2), Fr(3, 2), Fr(3, 4)*24*(3 + 1)), (Fr(1, 2), Fr(1, 2), Fr(1, 4)*24*(3 + 1))],
        4: [(2, 0, 0), (Fr(1, 3), Fr(5, 3), 24*(Fr(1, 3)*3 + Fr(2, 3)*4)), (Fr(2, 3), Fr(4, 3), Fr(4, 3)*24*3),
            (Fr(1, 3), Fr(1, 3), 24*(Fr(1, 6)*3 + Fr(1, 6)*4)), (Fr(2, 3), Fr(2, 3), Fr(2, 3)*24*3),
            (Fr(1, 3), Fr(1, 3), 24*(Fr(1, 3)*3 + Fr(2, 3)*2))]}
QH = {2: 2, 3: 3, 4: 4}
def B0(D): r = sqrt(D + 1); return 3*(r + 1)*(sqrt(2*D + 2) + 1)**4
def Bm(D): r = sqrt(D + 1); return r + 1
def Bc(N, D):
    """bound for every c*(T) with discriminant D (all cusps), D >= -1:  2 max_F (|A| B0 + (|h0| + K_h U^2) Bm) e^{pi sqrt(D+1)}."""
    U = QH[N]*(D + 1)/4
    v = max(float(abs(A))*B0(D) + (float(abs(h0)) + float(K)*U*U)*Bm(D) for A, h0, K in BASE[N])
    return 2*v*exp(pi*sqrt(D + 1))
def Bc_logslope(N, x):
    """upper bound for d/dx log Bc at D = x^2 (x >= 10): degree-5 polynomial times e^{pi x}."""
    return pi + 5.0/x*1.05
# ---- c*(T) tables ----
def load(N):
    fn = os.path.join(HERE, 'cstar_N%d_%d.pkl' % (N, DEX))
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    G = pickle.load(open(os.path.join(HERE, 'gen_N%d_%d.pkl' % (N, DEX)), 'rb'))
    QD = G['QD']; C = G['C']
    Es = np.arange(-QD, QD*DEX + 1)
    out = {'QD': QD}
    for name, d in CUSPS[N].items():
        c, dl, Nx = d['c'], d['dl'], d['Nx']
        tabs = {}
        for r in sorted(set((c*dp) % N for dp in range(1, N+1))):
            for u in range(Nx):
                K = [k for k in range(N) if (Fr(c*k, N) - Fr(u, Nx)).denominator == 1]
                arr = np.zeros(len(Es))
                for i, E in enumerate(Es):
                    vals = [C[(r, k)].get(int(E), 0) for k in K]
                    if not any(vals): continue
                    best = 0.0
                    for ap in range(N):
                        z = sum(float(v)*cmath.exp(2j*pi*dl*ap*k/N) for v, k in zip(vals, K))
                        best = max(best, abs(z))
                    arr[i] = best
                tabs[(r, u)] = arr
        out[name] = tabs
        out[name + '_bar'] = np.max(np.array(list(tabs.values())), axis=0)
    # also exact |C| maxima check vs Bc
    pickle.dump(out, open(fn, 'wb'))
    return out
if __name__ == '__main__':
    import sys
    N = int(sys.argv[1])
    T = load(N)
    QD = T['QD']
    for name in CUSPS[N]:
        bar = T[name + '_bar']
        Ds = np.arange(-QD, QD*DEX+1)/QD
        ratio = max(bar[i]/Bc(N, Ds[i]) for i in range(0, len(Ds), 7))
        print(N, name, 'tables', sorted(T[name].keys()), 'c*(D=-1)', bar[0], 'max c*/Bc', ratio,
              'D=-1 entries by (r,u):', {k: v[0] for k, v in T[name].items()})
