# Weak Jacobi forms phi_{0,1} and phi_{-2,1}: coefficients c0(D), cm2(D), -1 <= D <= Dmax, from theta functions.
import flint
flint.ctx.cap = 400000
from flint import fmpz_series
from math import isqrt

def sigma1(n): return sum(d for d in range(1, n+1) if n % d == 0)

def weak_index1(Dmax):
    """c0[D] (phi_{0,1}) and cm2[D] (phi_{-2,1}) for -1<=D<=Dmax."""
    Nn = Dmax//4 + 2          # need q^n with n up to Nn
    P = 8*Nn + 16             # precision in units of q^{1/8}
    def ser(d): 
        c = [0]*P
        for e, v in d.items():
            if e < P: c[e] += v
        return fmpz_series(c, prec=P)
    R = isqrt(P) + 2
    # theta_i(z)^2 components y^J (J=0,1), theta_i(0)^2
    comp = {}
    for name in ['2', '3', '4', 'm1']:
        for J in [0, 1]:
            d = {}
            for a in range(-R, R+1):
                if name in ('3', '4'):
                    b = J - a; e = 4*(a*a + b*b); sg = (1 if (a+b)%2==0 else -1) if name == '4' else 1
                elif name == '2':   # (a+1/2)+(b+1/2) = J  -> b = J-1-a
                    b = J - 1 - a; e = 4*(a*a + a + b*b + b) + 2; sg = 1
                else:               # -theta1^2 component y^J: sum (-1)^{a+b} q^{...} y^{a+b+1}
                    b = J - 1 - a; e = 4*(a*a + a + b*b + b) + 2; sg = (1 if (a+b)%2==0 else -1)
                if e < P: d[e] = d.get(e, 0) + sg
            comp[(name, J)] = d
    zero = {}
    for name in ['2', '3', '4']:
        d = {}
        for a in range(-R, R+1):
            for b in range(-R, R+1):
                if name == '2': e = 4*(a*a + a + b*b + b) + 2; sg = 1
                else: e = 4*(a*a + b*b); sg = (1 if (a+b)%2==0 else -1) if name == '4' else 1
                if e < P: d[e] = d.get(e, 0) + sg
        zero[name] = d
    out0, outm2 = {}, {}
    # eta^6 = q^{1/4} prod (1-q^k)^6 : in eighths shift 2
    pe = [0]*P; pe[0] = 1
    for k in range(1, P//8 + 1):
        for _ in range(6):
            for i in range(P-1, 8*k-1, -1): pe[i] -= pe[i-8*k]
    inv_eta6 = 1/fmpz_series(pe, prec=P)
    for J in [0, 1]:
        tot = fmpz_series([0], prec=P)
        for name in ['3', '4']:
            tot += 4*ser(comp[(name, J)]) * (1/ser(zero[name]))
        # theta2: both carry factor q^{1/4}: remove shift 2 from each
        num = {e-2: v for e, v in comp[('2', J)].items()}
        den = {e-2: v for e, v in zero['2'].items()}
        # 4*theta2(z)^2/theta2(0)^2 ; den has constant 4
        t2 = ser(num) * (1/ (ser(den)/4)) if False else None
        dens = ser({e: v//4 for e, v in den.items()}) if all(v % 4 == 0 for v in den.values()) else None
        tot += ser(num) * (1/dens)
        cl = tot.coeffs()
        m2 = ser({e-2: v for e, v in comp[('m1', J)].items()}) * inv_eta6
        ml = m2.coeffs()
        for e in range(0, P - 16):
            if e % 8: 
                assert (cl[e] if e < len(cl) else 0) == 0 and (ml[e] if e < len(ml) else 0) == 0, e
                continue
            n = e//8; D = 4*n - J*J
            if D <= Dmax:
                out0[D] = int(cl[e]) if e < len(cl) else 0
                outm2[D] = int(ml[e]) if e < len(ml) else 0
    return out0, outm2

