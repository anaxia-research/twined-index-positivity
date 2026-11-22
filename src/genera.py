# Twisted-twining elliptic genera phi_{(r,s)} of K3 for g = 2A, 3A, 4B and their eigen-projections C_{r,k}(D).
# phi_{(r,s)} = A_{r,s} phi_{0,1} + h_{(r,s)}(tau) phi_{-2,1};  coefficients depend on D = 4n - l^2 only.
# Tables are keyed by E = QD*D (QD = 3 for N=3, else 1).
import sys, os, pickle
from fractions import Fraction as Fr
from math import isqrt
import flint
from flint import fmpz_poly

HERE = os.path.dirname(os.path.abspath(__file__))

def sig1_table(n):
    s = [0]*(n+1)
    for d in range(1, n+1):
        for k in range(d, n+1, d): s[k] += d
    return s

def weak(Dmax):
    fn = os.path.join(HERE, 'weak_%d.pkl' % Dmax)
    if os.path.exists(fn): return pickle.load(open(fn, 'rb'))
    from weak import weak_index1
    c0, cm2 = weak_index1(Dmax)
    pickle.dump((c0, cm2), open(fn, 'wb'))
    return c0, cm2

class Series:
    """q-series sum_u a_u q^{u/Q} with Fraction coefficients, u in [0, P)."""
    def __init__(s, Q, P, a=None): s.Q, s.P = Q, P; s.a = a if a is not None else [Fr(0)]*P
    def __add__(s, o): return Series(s.Q, s.P, [x + y for x, y in zip(s.a, o.a)])
    def scale(s, c): return Series(s.Q, s.P, [c*x for x in s.a])

def E2(Q, P, a, b2, d, S1):
    """E2((a tau + b)/d) with b/d = b2/2 (b2 in {0,1}): 1 - 24 sum sigma1(n) (-1)^{b2 n} q^{a n/d}; units q^{1/Q}."""
    out = [Fr(0)]*P; out[0] = Fr(1)
    # exponent a n/d = u/Q  ->  u = Q a n / d
    assert (Q*a) % d == 0
    st = Q*a//d
    n = 1
    while n*st < P:
        out[n*st] += -24*S1[n]*((-1)**(n*b2))
        n += 1
    return Series(Q, P, out)

def phi2(Q, P, M, a, d, S1):
    """phi_2^{(M)}((a/d) tau) = (M E2(M a tau/d) - E2(a tau/d))/(M-1)."""
    from math import gcd
    g1 = gcd(M*a, d); g2 = gcd(a, d)
    return (E2(Q, P, M*a//g1, 0, d//g1, S1).scale(Fr(M)) + E2(Q, P, a//g2, 0, d//g2, S1).scale(Fr(-1))).scale(Fr(1, M-1))

def sectors(N, P, S1):
    """dict (r,s) -> (A, h Series in q^{1/N}) for the representatives needed."""
    Q = N
    one = Series(Q, P); one.a[0] = Fr(1)
    F = {}
    if N == 2:
        F[(0, 1)] = (Fr(2, 3), phi2(Q, P, 2, 1, 1, S1).scale(Fr(4, 3)))
        F[(1, 0)] = (Fr(2, 3), phi2(Q, P, 2, 1, 2, S1).scale(Fr(-2, 3)))
    elif N == 3:
        F[(0, 1)] = (Fr(1, 2), phi2(Q, P, 3, 1, 1, S1).scale(Fr(3, 2)))
        F[(1, 0)] = (Fr(1, 2), phi2(Q, P, 3, 1, 3, S1).scale(Fr(-1, 2)))
    elif N == 4:
        F[(0, 1)] = (Fr(1, 3), phi2(Q, P, 2, 1, 1, S1).scale(Fr(-1, 3)) + phi2(Q, P, 4, 1, 1, S1).scale(Fr(2)))
        F[(0, 2)] = (Fr(2, 3), phi2(Q, P, 2, 1, 1, S1).scale(Fr(4, 3)))
        F[(1, 0)] = (Fr(1, 3), phi2(Q, P, 2, 1, 2, S1).scale(Fr(1, 6)) + phi2(Q, P, 4, 1, 4, S1).scale(Fr(-1, 2)))
        F[(2, 0)] = (Fr(2, 3), phi2(Q, P, 2, 1, 2, S1).scale(Fr(-2, 3)))
        # (2,1) = (0,1)|[[1,0],[2,1]]:  -(1/3) e_2 + (2/3)(E2(tau-1/2) - E2(tau))
        e2 = phi2(Q, P, 2, 1, 1, S1)
        F[(2, 1)] = (Fr(1, 3), e2.scale(Fr(-1, 3)) + (E2(Q, P, 1, 1, 1, S1) + E2(Q, P, 1, 0, 1, S1).scale(Fr(-1))).scale(Fr(2, 3)))
    return F

def form_table(A, h, N, QD, Dmax, c0, cm2):
    """coefficients of A phi01 + h phi_{-2,1} at (n,l), n = v/Q mod 1, returned as dict (E, v) -> Fraction,
    E = QD*D, D = 4n - l^2 (the parity of l is fixed by (D, v))."""
    from math import gcd
    Q = h.Q
    den = 1
    for x in list(h.a) + [A]:
        if x: den = den*x.denominator//gcd(den, x.denominator)
    L = den
    nmax = Dmax//4 + 2
    out = {}
    for lam in (0, 1):
        c0l = [c0.get(4*n - lam, 0) for n in range(nmax+1)]
        cml = [cm2.get(4*n - lam, 0) for n in range(nmax+1)]
        Pm = fmpz_poly(cml)
        for v in range(Q):
            hv = []
            i = 0
            while v + Q*i < len(h.a) and i <= nmax:
                hv.append(int(h.a[v + Q*i]*L)); i += 1
            pr = (fmpz_poly(hv)*Pm).coeffs() if any(hv) else []
            for nn in range(nmax+1):
                val = int(pr[nn]) if nn < len(pr) else 0
                if v == 0: val += int(A*L)*c0l[nn]
                n = Fr(v, Q) + nn
                D = 4*n - lam
                if D > Dmax: continue
                E = D*QD
                assert E.denominator == 1
                out[(int(E), v)] = Fr(val, L)
    return out

def build(N, Dmax):
    QD = 3 if N == 3 else 1
    c0, cm2 = weak(Dmax)
    P = N*(Dmax//4 + 3)
    S1 = sig1_table(P + 2)
    F = sectors(N, P, S1)
    tab = {key: form_table(A, h, N, QD, Dmax, c0, cm2) for key, (A, h) in F.items()}
    tab[(0, 0)] = {(E, 0): Fr(2*c0.get(E//QD, 0)) for E in range(-QD, QD*Dmax+1, QD)}
    Q = N
    def get(t, E, v): return t.get((E, v), Fr(0))
    Es = range(-QD, QD*Dmax+1)
    # untwisted twining genera c_{g^s} (n integer, class v = 0)
    from math import gcd
    cg = {}
    for s in range(N):
        o = gcd(s, N)
        key = (0, 0) if s % N == 0 else ((0, 1) if o == 1 else (0, o))
        cg[s] = {E: get(tab[key], E, 0) for E in Es}
    C = {}
    for k in range(N):
        C[(0, k)] = {E: sum(cg[s][E]*cos_rational(Fr(-s*k, N)) for s in range(N))/N for E in Es}
    if N in (2, 3):
        for r in range(1, N):
            for k in range(N):
                v = (r*k) % N          # n = rk/N mod 1, Q = N
                C[(r, k)] = {E: get(tab[(1, 0)], E, v) for E in Es}
    else:
        for r in (1, 3):
            for k in range(4):
                v = (r*k) % 4
                C[(r, k)] = {E: get(tab[(1, 0)], E, v) for E in Es}
        t20, t21 = tab[(2, 0)], tab[(2, 1)]
        assert all(t21[key] == 0 for key in t21 if key[1] in (1, 2, 3))
        C[(2, 0)] = {E: (get(t20, E, 0) + get(t21, E, 0))/2 for E in Es}
        C[(2, 2)] = {E: (get(t20, E, 0) - get(t21, E, 0))/2 for E in Es}
        C[(2, 1)] = {E: get(t20, E, 2)/2 for E in Es}
        C[(2, 3)] = dict(C[(2, 1)])
        assert all(t20[key] == 0 for key in t20 if key[1] in (1, 3))
    return dict(N=N, QD=QD, Dmax=Dmax, tab=tab, cg=cg, C=C)

def cos_rational(fr):
    """cos(2 pi fr) for fr with denominator 1,2,3,4,6 -- exact."""
    x = fr - (fr.numerator//fr.denominator)
    table = {Fr(0): Fr(1), Fr(1, 2): Fr(-1), Fr(1, 4): Fr(0), Fr(3, 4): Fr(0), Fr(1, 3): Fr(-1, 2), Fr(2, 3): Fr(-1, 2),
             Fr(1, 6): Fr(1, 2), Fr(5, 6): Fr(1, 2)}
    return table[x]

if __name__ == '__main__':
    N, Dmax = int(sys.argv[1]), int(sys.argv[2])
    res = build(N, Dmax)
    # integrality and D=-1 checks
    QD = res['QD']
    for key, t in res['C'].items():
        bad = [E for E, v in t.items() if v.denominator != 1]
        print('C', key, 'non-integral:', len(bad), bad[:5], ' C(D=-1) =', t.get(-QD), ' first:', [(Fr(E, QD), int(v)) for E, v in sorted(t.items()) if v][:6])
    for s, t in res['cg'].items():
        print('c_{g^%d}' % s, [(Fr(E, QD), t[E]) for E in sorted(t) if t[E] and E <= 8*QD])
    for key, t in res['tab'].items():
        print('phi', key, [(Fr(E, QD), v, t[(E, v)]) for (E, v) in sorted(t) if t[(E, v)] and E <= 4*QD])
    pickle.dump(res, open(os.path.join(HERE, 'gen_N%d_%d.pkl' % (N, Dmax)), 'wb'))
