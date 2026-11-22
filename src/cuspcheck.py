# Numerical check of the cusp expansion of L*W_L|gamma (twisted-sector formula) against direct evaluation.
import sys, pickle, mpmath as mp
from math import isqrt
from fractions import Fraction as Fr
mp.mp.dps = 60
N = int(sys.argv[1]); G = pickle.load(open('gen_N%d_2400.pkl' % N, 'rb'))
QD = G['QD']; cg = G['cg']; C = G['C']
def e(x): return mp.exp(2j*mp.pi*x)
def LW_inf(L, s, v, kmax):
    tot = mp.mpc(0)
    for t in range(1, L+1):
        if L % t: continue
        Lt = L//t
        for k in range(0, kmax//t + 1):
            jm = isqrt(4*k*Lt + 1)
            for j in range(-jm, jm+1):
                D = 4*k*Lt - j*j
                if D < -1: continue
                c = cg[t % N].get(QD*D, 0)
                if c: tot += Lt*int(c)*e(k*t*s + j*t*v)
    return tot
def LW_cusp(L, s, v, c, dl, xmax):
    tot = mp.mpc(0)
    for ap in range(1, L+1):
        if L % ap: continue
        dp = L//ap; r = (c*dp) % N
        for k in range(N):
            ph = e(Fr(dl*ap*k, N))
            x0 = Fr(c*k, N) - (Fr(c*k, N).numerator//Fr(c*k, N).denominator)
            x = x0
            while x <= xmax:
                jm = isqrt(int(4*dp*x) + 1) + 1
                for l in range(-jm, jm+1):
                    D = 4*dp*x - l*l
                    if D < -1: continue
                    E = D*QD; assert E.denominator == 1
                    val = C[(r, k)].get(int(E), 0)
                    if val: tot += dp*ph*int(val)*e(ap*x*s + ap*l*v)
                x += 1
    return tot
def slash(L, gam, s, v, kmax):
    a, b, c, d = gam
    j = c*s + d
    return e(-L*c*v*v/j)*LW_inf(L, (a*s + b)/j, v/j, kmax)
tests = {2: [((0, -1, 1, 0), 1, 0, mp.mpc(0.05, 1.0))],
         3: [((0, -1, 1, 0), 1, 0, mp.mpc(0.05, 1.0))],
         4: [((0, -1, 1, 0), 1, 0, mp.mpc(0.05, 1.0)), ((1, 0, 2, 1), 2, 1, mp.mpc(-0.47, 0.5))]}
v = mp.mpc(0.11, 0.03)
for gam, c, dl, s in tests[N]:
    for L in range(1, 5):
        A = slash(L, gam, s, v, 90); B = LW_cusp(L, s, v, c, dl, 90)
        print(N, gam, L, mp.nstr(A, 15), mp.nstr(B, 15), 'rel err', mp.nstr(abs(A-B)/abs(A), 3))
