# Lower bounds for the untwined index d*(m,n,l), 0 <= l <= m <= n, from Rossello's Rademacher expansion.
import mpmath as mp
from math import log, sqrt, pi
from psig import psi_coeffs
mp.mp.dps = 30
def dcoef(N):
    """coefficients d(n) of 1/eta^24 = sum_{n >= -1} d(n) q^n."""
    c = [0]*(N+2); c[0] = 1
    for k in range(1, N+2):
        for _ in range(24):
            for i in range(k, N+2): c[i] += c[i-k]
    return {n: c[n+1] for n in range(-1, N+1)}
_polar = {}
def polar(m):
    """polar coefficients c_m(n~, l~) (4 m n~ - l~^2 < 0, 0 <= l~ <= m) of psi_m for 1/Phi_10."""
    if m in _polar: return _polar[m]
    nq = m//4 + 1
    r = psi_coeffs(m, nq, 1)
    P = {}
    for nt in range(-1, nq+1):
        for lt in range(0, m+1):
            if 4*m*nt - lt*lt < 0: P[(nt, lt)] = r[(m, nt, lt)]
    _polar[m] = P
    return P
def logCpos(m, D):
    C = 2*mp.pi*(m+2)/mp.sqrt(2*m)*((4*m + m*m)/mp.mpf(D))**(mp.mpf(23)/4)*mp.besseli(mp.mpf(23)/2, 2*mp.pi*mp.sqrt(mp.mpf(D)/m*(1 + mp.mpf(m)/4)))
    return float(mp.log(C))
def lam_small(m, Da):
    """lambda_m(Da): for D >= Da, d*(m,n,l) >= lambda * Cpos(D,m,-1)  (m <= 5)."""
    P = polar(m); d = dcoef(m+1)
    b = mp.pi/m*mp.sqrt(m*m + 4*m); x = mp.sqrt(Da)
    neg = 0
    for (nt, lt), c in P.items():
        if lt == m: continue
        Dt = abs(4*m*nt - lt*lt); a = mp.pi/m*mp.sqrt(Dt)
        neg += 2*abs(c)/(m+2)*(mp.mpf(Dt)/(m*m + 4*m))**(mp.mpf(23)/4)*mp.exp((a - b)*x)*mp.sqrt(b/a)
    Cnorm = 2*mp.pi*(m+2)/mp.sqrt(2*m)*(mp.mpf(m*m + 4*m))**(mp.mpf(23)/4)
    a12 = 2*mp.pi/mp.sqrt(m)
    e12 = mp.mpf(23)/21*mp.sqrt(2*m)*d[m]*(4*m)**6*mp.mpf(Da)**(-6)/(Cnorm*mp.mpf(Da)**(-mp.mpf(23)/4))*mp.exp((a12 - b)*x)*mp.sqrt(b/a12)
    e25 = mp.mpf(23)/21*d[m]*(4*m)**(mp.mpf(25)/4)*m**1.5*mp.pi*(mp.pi**2/3 + mp.mpf(1)/3)*mp.mpf(Da)**(-mp.mpf(25)/4)/(Cnorm*mp.mpf(Da)**(-mp.mpf(23)/4))*mp.exp((a12 - b)*x)*mp.sqrt(b/a12)
    return float(mp.mpf(19)/21 - mp.mpf(23)/21*neg - e12 - e25)
def Lam_asym(m, cm, xK):
    """lowL(x) >= Ax x - 12 log x + Lam for x >= xK, using that sqrt(y) e^{-y} I_{23/2}(y) increases."""
    yK = pi*xK*sqrt(1 + 4/m)
    logK = float(mp.log(mp.sqrt(yK)*mp.exp(-yK)*mp.besseli(mp.mpf(23)/2, yK)))
    Lam = log(cm) + log(2*pi*(m+2)/sqrt(2*m)) + 23/4*log(m*m + 4*m) - 0.5*log(pi*sqrt(1 + 4/m)) + logK
    return pi*sqrt(1 + 4/m), Lam
