# Upper bound U_N(m, D) for log |d_{g}(m,n,l)| (twined R-chamber coefficient, g = 2A, 3A, 4B), from the box tables.
import numpy as np, pickle
from math import sqrt, pi, log, exp
from cusps import CUSPS, kw as KW
from tbox import kap, s_c
G2 = 2*log(2) - 2*log(1 - exp(-2*pi))
def logprod(x, N=80):
    """lower bound for sum_{n>=1} log(1 - x^n), 0<x<1."""
    x = np.asarray(x, float); tot = np.zeros_like(x)
    for n in range(1, N+1): tot += np.log1p(-x**n)
    xn = x**(N+1)
    return tot - xn/((1-x)*(1-xn))
def logprod_half(x, N=80):
    """lower bound for sum_{n>=1} log(1 - x^(n-1/2))."""
    x = np.asarray(x, float); tot = np.zeros_like(x)
    for n in range(1, N+1): tot += np.log1p(-x**(n-0.5))
    xn = x**(N+0.5)
    return tot - xn/((1-x)*(1-xn))
def upper_log1p_sum(x): x = np.asarray(x, float); return x/(1-x)
def log_sinh(u):
    """log sinh(u) for u >= 0 without overflow (-inf at u = 0)."""
    u = np.asarray(u, float)
    with np.errstate(divide='ignore'):
        return u - log(2) + np.log1p(-np.exp(-2*u))
def cfix(cu, kw, s):
    """upper bound for -kw log s - log|eta_g|gamma| - 2 pi w s - log|phi_{-2,1}| - hlow  (s-monotone parts), at s."""
    s = np.asarray(s, float); x = np.exp(-2*pi*s)
    logC = cu['lc'] + sum(m*logprod(np.exp(-2*pi*r*s)) for r, m in cu['eta'])
    return -kw*np.log(s) - logC - log(4) - 4*logprod_half(x) + 4*upper_log1p_sum(x)
class Bound:
    def __init__(self, N, fn, smax_global=1e4, ratio_large=1.01, t_ref=0.2):
        tb = pickle.load(open(fn, 'rb')); assert tb['N'] == N
        self.N = N; self.kw = KW(N); self.rho0, self.eta1 = tb['rho0'], tb['eta1']; self.t_ref = t_ref
        self.C = {}; self.P = {}
        for name, cu in CUSPS[N].items():
            sg, bg = tb[name]['sg'], tb[name]['bg']; nb = len(bg) - 1; last = len(sg) - 2
            rows = [(sg[i], sg[i+1], bg[jb], bg[jb+1], r['total']) for (i, jb), r in tb[name]['box'].items()]
            s = sg[-1]; edges = [s]
            while s < smax_global: s *= ratio_large; edges.append(s)
            for a, b in zip(edges[:-1], edges[1:]):
                for jb in range(nb): rows.append((a, b, bg[jb], bg[jb+1], tb[name]['box'][(last, jb)]['total']))
            R = np.array(rows)
            d = dict(s0=R[:, 0], s1=R[:, 1], b0=R[:, 2], b1=R[:, 3], tail=R[:, 4])
            d['Gfix'] = cfix(cu, self.kw, d['s0'])
            self.C[name] = d
            # analytic data: grid boxes only
            G = np.array([(sg[i], sg[i+1], bg[jb], bg[jb+1], r['total']) for (i, jb), r in tb[name]['box'].items()])
            s0, s1, b0, b1, tl = G.T
            a = cu['A']
            kk = np.maximum(np.maximum(a*s0 + self.eta1, self.rho0/s0), np.maximum(a*s1 + self.eta1, self.rho0/s1))
            base = tl + 2*pi*b1*b1*s1 + 2*pi*cu['w']*s1 + cfix(cu, self.kw, s0)
            Tend = max(tb[name]['box'][(last, jb)]['total'] for jb in range(nb))
            self.P[name] = dict(s0=s0, b0=b0, kk=kk, base=base, send=sg[-1], Tend=Tend, cend=float(cfix(cu, self.kw, sg[-1])))
    def supF(self, name, m, t, a0):
        cu = CUSPS[self.N][name]; d = self.C[name]
        smax = 1/(cu['cmin']**2*t)
        if smax < sqrt(3)/2: return -np.inf
        ok = d['s0'] < smax
        s0 = d['s0'][ok]; s1 = np.minimum(d['s1'][ok], smax); b0 = d['b0'][ok]; b1 = d['b1'][ok]
        a = cu['A']; rho0, eta1 = self.rho0, self.eta1
        kk = np.maximum(np.maximum(a*s0 + eta1, rho0/s0), np.maximum(a*s1 + eta1, rho0/s1))
        x = np.maximum(a0*np.sqrt(t*s0), b0*s0)
        hlow = np.maximum(np.log(4*x*x), 2*log_sinh(pi*b0*s0))
        G = 2*pi*b1*b1*s1 + 2*pi*cu['w']*s1 + d['Gfix'][ok] - hlow
        return float(np.max(2*pi*(m+1)*kk + d['tail'][ok] + G))
    def table(self, m, TS, AST):
        """S[i,k] = kw log t_i + max_cusp supF(m, t_i, a_k)."""
        S = np.zeros((len(TS), len(AST)))
        for i, t in enumerate(TS):
            for k, a in enumerate(AST):
                S[i, k] = self.kw*log(t) + max(self.supF(nm, m, t, a) for nm in CUSPS[self.N])
        return S
    # ---- analytic (large D) branches ----
    def SFgrid(self, name, m, a0):
        d = self.P[name]
        x = np.maximum(a0*np.sqrt(self.t_ref*d['s0']), d['b0']*d['s0'])
        h = np.maximum(np.log(4*x*x), 2*log_sinh(pi*d['b0']*d['s0']))
        return float(np.max(2*pi*(m+1)*d['kk'] + d['base'] - h))
    def branches(self, m, tau, a0, xK):
        """U(x) <= max_b (CX x + CL log x + C0 + CI/x) for x >= xK, with t = tau/x <= t_ref."""
        assert tau/xK <= self.t_ref
        bx, bl, b0, bi = pi*tau/(2*m), -self.kw, self.kw*log(tau), 2*pi*tau*m*a0*a0
        out = []
        for name, cu in CUSPS[self.N].items():
            d = self.P[name]
            out.append((name+':G', bx, bl + 1.0, b0 + self.SFgrid(name, m, a0) + log(self.t_ref/tau), bi))
            ks = kap(cu, d['send'], self.rho0, self.eta1)
            KE = 2*pi*(m+1)*ks + 2*pi*cu['w']*d['send'] + d['cend'] + d['Tend']
            g1c = 2*pi/d['send'] - log(4*a0*a0*tau*d['send'])
            out.append((name+':E', bx, bl, b0 + KE + G2, bi)); out.append((name+':Ep', bx, bl + 1.0, b0 + KE + g1c, bi))
            sig = 1/(cu['cmin']**2*tau)
            assert sig*xK >= max(d['send'], s_c(cu, self.rho0, self.eta1))
            # at s = sig x : kappa = A s + eta1 ; cend at send is an upper bound for cfix(s) + kw log s - kw log send ... (handled below)
            cxL = 2*pi*(m+1)*cu['A']*sig + 2*pi*cu['w']*sig
            c0L = 2*pi*(m+1)*self.eta1 + d['cend'] + self.kw*log(d['send']) - self.kw*log(sig) + d['Tend']
            out.append((name+':L', bx + cxL, bl - self.kw, b0 + c0L + G2, bi))
            out.append((name+':Lp', bx + cxL, bl - self.kw + 1.0, b0 + c0L + g1c, bi))
        return out
