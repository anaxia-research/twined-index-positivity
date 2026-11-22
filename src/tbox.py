# Box bounds for the tail of the Siegel-type product at every cusp of Gamma_0(N), N = 2, 3, 4.
# Point (s, beta): s = Im sigma' >= sqrt3/2, beta = |Im v'|/s in [0,1/2]. Cauchy radius R = beta^2 s + kappa(s),
# kappa(s) = max(A s + eta1, rho0/s), A = 1/(4 Nc^2). Term T = (l, x, j), x in (1/Nx) Z:
#   D = 4 l x - j^2,  tr = l kappa(s) + s q,  q = l (beta + j/(2l))^2 + D/(4l).
import numpy as np, pickle, sys, time
from math import sqrt, pi, log, exp, floor, ceil
from cusps import CUSPS, load, Bc, DEX
def kap(cu, s, rho0, eta1): return max(cu['A']*s + eta1, rho0/s)
def s_c(cu, rho0, eta1):
    a = cu['A']; return (-eta1 + sqrt(eta1**2 + 4*a*rho0))/(2*a)
def min_g(cu, l, q, s0, s1, rho0, eta1):
    """exact min over s in [s0,s1] of l*kappa(s) + s*q (convex in s)."""
    a = cu['A']; sc = s_c(cu, rho0, eta1)
    def g(s): return l*np.maximum(a*s + eta1, rho0/s) + s*q
    out = np.minimum(g(s0), g(s1))
    if s0 < sc < s1: out = np.minimum(out, g(sc))
    hi = min(s1, sc)
    with np.errstate(divide='ignore', invalid='ignore'):
        st = np.where(q > 0, np.sqrt(l*rho0/np.where(q > 0, q, 1.0)), -1.0)
    ok = (st > s0) & (st < hi)
    if ok.any(): out = np.where(ok, np.minimum(out, g(np.where(ok, st, s0))), out)
    return out
class Tab:
    def __init__(self, N):
        self.N = N; T = load(N); self.QD = T['QD']; self.T = T
        self.stack = {}
        for name, cu in CUSPS[N].items():
            Nx = cu['Nx']
            st = {}
            for r in sorted(set(k[0] for k in T[name])):
                st[r] = np.stack([T[name][(r, u)] for u in range(Nx)])
            self.stack[name] = st
def tail_box(TB, name, s0, s1, b0, b1, rho0, eta1, X1, theta):
    N = TB.N; cu = CUSPS[N][name]; QD = TB.QD; Nx, c, Nc = cu['Nx'], cu['c'], cu['Nc']
    sc = s_c(cu, rho0, eta1)
    kmin = kap(cu, sc, rho0, eta1) if s0 < sc < s1 else min(kap(cu, s0, rho0, eta1), kap(cu, s1, rho0, eta1))
    tot = 0.0; nterms = 0; mintr = 1e9; maxD = -2
    Lmax = int(floor((X1 + s1/4)/kmin)) + 1
    st = TB.stack[name]
    for l in range(1, Lmax+1):
        r = (c*l) % N
        tab = st[r]
        rem = max(0.0, X1 - l*kmin)
        rr = sqrt(rem/(s0*l) + 1/(4*l*l))
        jlo = int(ceil(-2*l*(b1 + rr))) - 1; jhi = int(floor(-2*l*(b0 - rr))) + 1
        j = np.arange(jlo, jhi+1, dtype=np.int64)
        Dmax = 4*l*rem/s0 + 1
        umin = np.ceil(Nx*(j*j - 1)/(4.0*l)).astype(np.int64)
        umax = np.floor(Nx*(Dmax + j*j)/(4.0*l)).astype(np.int64)
        nU = umax - umin + 1
        if nU.max() <= 0: continue
        i = np.arange(int(nU.max()), dtype=np.int64)
        u = umin[:, None] + i[None, :]
        mask = u <= umax[:, None]
        num = QD*4*l*u
        assert (num % Nx == 0).all()
        E = num//Nx - QD*(j*j)[:, None]
        mask &= E >= -QD
        D = E/QD
        x = -j/(2.0*l)
        dist = np.maximum(0.0, np.maximum(b0 - x, x - b1))
        q = l*dist[:, None]**2 + D/(4.0*l)
        tr = min_g(cu, l, q, s0, s1, rho0, eta1)
        sel = mask & (tr <= X1)
        if not sel.any(): continue
        Ec = np.where(sel, E, 0)
        call = tab[np.where(sel, u, 0) % Nx, Ec + QD]
        sel &= ~((E < 0) & (call == 0))        # D = -1 terms with zero coefficient (twisted sectors, k != 0)
        if not sel.any(): continue
        assert (tr[sel] > 0).all(), (name, l, s0, s1, b0, b1, tr[sel].min())
        Es = E[sel]; us = u[sel]; trs = tr[sel]
        assert Es.max() <= QD*DEX
        cst = tab[us % Nx, Es + QD]
        assert ((Es >= 0) | (l % Nc == 0)).all()
        tot += float(np.sum(cst*(-np.log1p(-np.exp(-2*pi*trs)))))
        nterms += int(sel.sum()); mintr = min(mintr, float(trs.min())); maxD = max(maxD, float(Es.max())/QD)
    # ---- remainder: D >= 0 terms with tr > X1 ----
    rho = kap(cu, s0, rho0, eta1)*s0                 # min det Y over the box (kappa(s) s increasing)
    bar = TB.T[name + '_bar']
    Es = np.arange(0, QD*DEX + 1); Ds = Es/QD
    K1 = float(np.max(bar[Es + QD]*np.exp(-2*pi*(1-theta)*np.maximum(X1, np.sqrt(rho*Ds)))))
    x = sqrt(DEX)
    assert 5.25/x < pi*(2*(1-theta)*sqrt(rho) - 1), ('theta too big', rho, theta)
    K2 = Bc(N, DEX)*exp(-2*pi*(1-theta)*max(X1, sqrt(rho*DEX)))
    Kpp = max(K1, K2)
    Aq = exp(-2*pi*theta*kmin)
    lam = 0.0
    for l in range(1, 4000): lam += Aq**l*(2*l*(b1-b0) + 3 + sqrt(2*l/(theta*s0)))
    lt = 4000; lam += Aq**lt*(2*lt*(b1-b0) + 3 + sqrt(2*lt/(theta*s0)))*4/(1-Aq)**2
    lam /= (1 - exp(-2*pi*theta*s0/Nx))
    rem_psd = Kpp*lam/(1 - exp(-2*pi*X1))
    # ---- remainder: D = -1 terms (only l in Nc Z) with tr > X1:  tr >= l eta1 + s0 (l A - 1/(4l) + l dist^2) ----
    cm1 = float(bar[0])
    lamm = 0.0
    for l in range(Nc, 40000, Nc):
        cl = l*cu['A'] - 1/(4*l)
        assert cl >= -1e-15
        lamm += exp(-pi*(l*eta1 + s0*cl))*(2*l*(b1-b0) + 3 + sqrt(4*l/s0))
    rem_m1 = cm1*exp(-pi*X1)*lamm/(1 - exp(-2*pi*X1))
    return dict(enum=tot, rem_psd=rem_psd, rem_m1=rem_m1, kmin=kmin, rho=rho, n=nterms, mintr=mintr, maxD=maxD,
                total=(tot + rem_psd + rem_m1)*(1 + 1e-9) + 1e-9)
def grids(cu, rho0, eta1, ratio, nb):
    sc = s_c(cu, rho0, eta1)
    sg = [sqrt(3)/2]
    while sg[-1] <= sc: sg.append(sg[-1]*ratio)
    return sg, list(np.linspace(0, 0.5, nb+1))
if __name__ == '__main__':
    N = int(sys.argv[1]); rho0, eta1, X1, theta, ratio, nb = [float(x) for x in sys.argv[2:8]]; nb = int(nb)
    TB = Tab(N)
    out = {'N': N, 'rho0': rho0, 'eta1': eta1, 'X1': X1, 'theta': theta}
    for name, cu in CUSPS[N].items():
        sg, bg = grids(cu, rho0, eta1, ratio, nb)
        out[name] = {'sg': sg, 'bg': bg, 'box': {}}
        t0 = time.time()
        for i in range(len(sg)-1):
            for jb in range(nb):
                out[name]['box'][(i, jb)] = tail_box(TB, name, sg[i], sg[i+1], bg[jb], bg[jb+1], rho0, eta1, X1, theta)
        vals = out[name]['box'].values()
        print(N, name, 'boxes', len(out[name]['box']), 's_end %.4f' % sg[-1], 'max total %.4f' % max(v['total'] for v in vals),
              'max rem %.2e' % max(v['rem_psd'] + v['rem_m1'] for v in vals), 'min tr %.4f' % min(v['mintr'] for v in vals),
              'maxD', max(v['maxD'] for v in vals), 'terms', sum(v['n'] for v in vals), 'time %.1f' % (time.time()-t0), flush=True)
    pickle.dump(out, open('tb_N%d_%g_%g_%g.pkl' % (N, rho0, eta1, X1), 'wb'))
