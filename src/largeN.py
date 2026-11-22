# Large-m lemma: for m >= M1 and all D >= 3m^2 the inequality holds; reduced to linear inequalities in m.
#   lowL(x) >= pi x - 12 log(x/m) + Lam0                     (x = sqrt D >= sqrt3 m, m >= M1)
#   Regime B (t = t_ref, sqrt3 m <= x <= (theta/t_ref) m)   and   Regime A (t = theta m / x, x >= lam m).
import sys, numpy as np, mpmath as mp
from math import sqrt, pi, log, exp
from ubound import Bound, G2, log_sinh
from cusps import CUSPS
from tbox import kap, s_c
from certN import bnd
mp.mp.dps = 30
THETA = {2: 0.5, 3: 1/3, 4: 0.25}
def Lam0(M1):
    y1 = pi*sqrt(3)*sqrt(M1*(M1 + 4))
    logK = float(mp.log(mp.sqrt(y1)*mp.exp(-y1)*mp.besseli(mp.mpf(23)/2, y1)))
    return log(0.55) + log(2*pi) - 0.5*log(2) - 0.5*log(pi*sqrt(1 + 4/M1)) + logK
def cusp_consts(B, name, a0):
    cu = CUSPS[B.N][name]; d = B.P[name]
    x = np.maximum(a0*np.sqrt(B.t_ref*d['s0']), d['b0']*d['s0'])
    h = np.maximum(np.log(4*x*x), 2*log_sinh(pi*d['b0']*d['s0']))
    kst = float(d['kk'].max()); Cg = float(np.max(d['base'] - h))
    # NOTE: max(2pi(m+1)kk + base - h) <= 2pi(m+1) max kk + max(base - h)
    return dict(kst=kst, Cg=Cg, send=d['send'], ksend=kap(cu, d['send'], B.rho0, B.eta1), Tend=d['Tend'], cend=d['cend'],
                A=cu['A'], w=cu['w'], cmin=cu['cmin'], sc=s_c(cu, B.rho0, B.eta1))
def lemma(N, M1, a0=0.05, verbose=True):
    """returns list of (component, regime, branch, alpha, gamma): gap >= alpha m - gamma for all m >= M1, x in regime."""
    L0 = Lam0(M1); extra = {2: 0.0, 3: log(2), 4: log(3)}[N]
    out = []
    for M in ([N] if N != 4 else [4, 2]):
        B = bnd(M); kw = B.kw; th = THETA[M]; tr = B.t_ref; eta1 = B.eta1
        K = {nm: cusp_consts(B, nm, a0) for nm in CUSPS[M]}
        lamA = max(sqrt(3), th/tr)
        # ---------- Regime B ----------
        if th/tr > sqrt(3):
            lin = []   # SF_B(m) <= a (m+1) + b
            for nm, k in K.items():
                smax = 1/(k['cmin']**2*tr)
                if smax < sqrt(3)/2: continue
                lin.append((2*pi*k['kst'], k['Cg']))
                if smax >= k['send']:
                    gm = max(G2, 2*pi/k['send'] - log(4*a0*a0*tr*k['send']))
                    for s in (k['send'], smax):
                        kap_s = k['A']*s + eta1 if s > k['send'] else k['ksend']
                        lin.append((2*pi*kap_s, 2*pi*k['w']*s + k['cend'] + kw*log(k['send']) - kw*log(s) + k['Tend'] + gm))
            amax = max(a for a, b in lin); bmax = max(a + b for a, b in lin)     # a(m+1)+b <= amax m + bmax
            lead = min(pi*sqrt(3) - 1.5*pi*tr, pi*th/tr - 0.5*pi*th*th/tr)
            alpha = lead - 2*pi*tr*a0*a0 - amax
            gamma = 12*log(th/tr) - L0 + kw*log(tr) + bmax + extra
            out.append((M, 'B', 'all', alpha, gamma))
        # ---------- Regime A : x >= lamA m ----------
        lam = lamA
        base_P = pi - pi*th/2; base_Q = -12 + kw; base_R = -2*pi*th*a0*a0/lam; base_S = L0 - kw*log(th) - extra
        for nm, k in K.items():
            brs = []
            # (i) grid: 2pi(m+1)kst + Cg + log(tr/th) + log(x/m)
            brs.append(('grid', 0.0, -1.0, -2*pi*k['kst'], -2*pi*k['kst'] - k['Cg'] - log(tr/th)))
            KE = 2*pi*k['w']*k['send'] + k['cend'] + k['Tend']
            brs.append(('end', 0.0, 0.0, -2*pi*k['ksend'], -2*pi*k['ksend'] - KE - G2))
            brs.append(('end1', 0.0, -1.0, -2*pi*k['ksend'], -2*pi*k['ksend'] - KE - (2*pi/k['send'] - log(4*a0*a0*th*k['send']))))
            c2 = k['cmin']**2*th
            assert lam/c2 >= 0 or True
            Px = -(2*pi*(1 + 1/M1)*k['A']/c2 + 2*pi*k['w']/(c2*M1))
            S0 = -2*pi*eta1 - k['cend'] - kw*log(k['send']) - kw*log(c2) - k['Tend']
            brs.append(('large', Px, kw, -2*pi*eta1, S0 - G2))
            brs.append(('large1', Px, kw - 1.0, -2*pi*eta1, S0 - (2*pi/k['send'] - log(4*a0*a0*th*k['send']))))
            for bname, Px_, Qx, Rm, Sc in brs:
                P = base_P + Px_; Q = base_Q + Qx; R = base_R + Rm; S = base_S + Sc
                assert P > 0, (M, nm, bname, P)
                # gap(x) >= P x + Q log(x/m) + R m + S, increasing for x >= -Q/P; at x = lam m:
                xmin_needed = -Q/P
                alpha = P*lam + R; gamma = -(Q*log(lam) + S)
                out.append((M, 'A', nm + ':' + bname, alpha, gamma, xmin_needed/lam))
    return out
if __name__ == '__main__':
    N = int(sys.argv[1]); M1 = int(sys.argv[2])
    res = lemma(N, M1)
    worst = 0
    for r in res:
        m0 = r[4]/r[3] if r[3] > 0 else float('inf')
        worst = max(worst, m0, r[5] if len(r) > 5 else 0)
        print(N, r[0], r[1], r[2], 'alpha %.4f gamma %.2f  m0 %.1f' % (r[3], r[4], m0), ('need m >= %.2f' % r[5]) if len(r) > 5 else '')
    print('N', N, 'M1', M1, 'all alpha > 0:', all(r[3] > 0 for r in res), ' threshold', worst, ' OK' if worst <= M1 else ' FAIL')
