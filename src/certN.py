# Per-m certification of positivity of the twined single-centred indices S_a for g = 2A (N=2), 3A (N=3), 4B (N=4):
#   N=2: d* > |d_2A|;  N=3: d* > 2|d_3A|;  N=4: d* > 2|d_4B| + |d_2A|,   for all D >= Dstart(m), 0 <= l <= m <= n.
import numpy as np, sys, pickle, time
from math import sqrt, pi, log, exp
from ubound import Bound
from cusps import CUSPS
from tbox import s_c
from lowbd import logCpos, lam_small, Lam_asym
MARGIN = 1e-3
TS = np.geomspace(0.002, 0.6, 70)
AST = (0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25)
TB = {2: 'tb_N2_0.35_0.05_30.pkl', 3: 'tb_N3_0.35_0.05_30.pkl', 4: 'tb_N4_0.35_0.05_30.pkl'}
BND = {}
def bnd(N):
    if N not in BND: BND[N] = Bound(N, TB[N])
    return BND[N]
def Ubest(m, D, S):
    slope = 2*pi*TS[:, None]/(4*m)
    U = slope*D + 2*pi*TS[:, None]*m*np.array(AST)[None, :]**2 + S
    i, k = np.unravel_index(np.argmin(U), U.shape)
    return float(U[i, k]), 2*pi*TS[i]/(4*m)
def rhs(N, m, D, S):
    u, sl = Ubest(m, D, S[N])
    if N == 2: return u, sl
    if N == 3: return log(2) + u, sl
    u2, sl2 = Ubest(m, D, S[2])
    return float(np.logaddexp(log(2) + u, u2)), max(sl, sl2)
def certify(N, m, Dstart, logLf, Dcap):
    S = {N: bnd(N).table(m, TS, AST)}
    if N == 4: S[2] = bnd(2).table(m, TS, AST)
    D = Dstart; steps = []; fails = []
    while D <= Dcap:
        lL = logLf(D); r, sl = rhs(N, m, D, S)
        g = lL - r
        if g <= MARGIN:
            fails.append((D, g)); D += 1
            if len(fails) > 2000: break
            continue
        Dn = max(D, int(np.floor(D + (g - MARGIN)/sl)))
        steps.append((D, Dn, g)); D = Dn + 1
    return steps, fails, D
def tau_of(N, m):
    return 2*sqrt(m*max(((m+1)*cu['A'] + cu['w'])/cu['cmin']**2 for cu in CUSPS[N].values()))
def asym_rig(N, m, Dk, cm, a0=0.25):
    xK = sqrt(Dk)
    Ax, Lam = Lam_asym(m, cm, xK)
    extra = {2: 0.0, 3: log(2), 4: log(3)}[N]
    br = []
    for M in ([N] if N != 4 else [4, 2]):
        tau = tau_of(M, m)
        br += [(M,) + b for b in bnd(M).branches(m, tau, a0, xK)]
    worst = None
    for M, name, CX, CL, C0, CI in br:
        slope = Ax - CX
        need = max(0.0, 12 + CL)/slope if slope > 0 else float('inf')
        g = (Ax*xK - 12*log(xK) + Lam) - (CX*xK + CL*log(xK) + C0 + CI/xK + extra)
        ok = slope > 0 and xK >= need and g > MARGIN
        if worst is None or g < worst[1]: worst = ((M, name), g, slope, need)
        if not ok: return False, ((M, name), g, slope, need, xK)
    return True, worst
if __name__ == '__main__':
    N = int(sys.argv[1])
    ms = range(int(sys.argv[2]), int(sys.argv[3]) + 1)
    Da = 200
    res = {}
    for m in ms:
        t0 = time.time()
        if m >= 6:
            cm = 0.55; Ds = 3*m*m
        else:
            cm = lam_small(m, Da); Ds = Da; assert cm > 0
        f = lambda D: log(cm) + logCpos(m, D)
        # certify up to Dcap, then the analytic tail from Dend
        Dcap = 400*m*m + 2000
        steps, fails, Dend = certify(N, m, Ds, f, Dcap)
        okA, infoA = asym_rig(N, m, Dend, cm)
        res[m] = dict(steps=steps, fails=fails, Dend=Dend, asym=(okA, infoA), cm=cm, Ds=Ds)
        print(N, m, 'steps', len(steps), 'fails', len(fails), fails[:3], 'Dend', Dend, 'asym', okA, infoA,
              'min gap', round(min(s[2] for s in steps), 3) if steps else None, 'cm', round(cm, 4), round(time.time()-t0, 1), flush=True)
    pickle.dump(res, open('certN%d_%s_%s.pkl' % (N, sys.argv[2], sys.argv[3]), 'wb'))
