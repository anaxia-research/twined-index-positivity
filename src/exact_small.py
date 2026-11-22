# Exact check for the charges not covered by the certificates: m <= 5, 3m^2 <= Delta < 200, 0 <= l <= m <= n.
# Checks S_a > 0 (all a) for Z2, Z3, Z4 twining, and integrality of S_a.
import pickle, sys
from fractions import Fraction as Fr
from psig import psi_coeffs, dstar
M, NQ = 5, 50
out = {}
for key, N, tw in (('1A', 1, 1), ('2A', 2, 1), ('3A', 3, 1), ('4B', 4, 1)):
    out[key] = psi_coeffs(M, NQ, N, twine=tw)
pickle.dump(out, open('exact_M5_NQ50.pkl', 'wb'))
cnt = 0; bad = []; minratio = {2: 1e9, 3: 1e9, 4: 1e9}
for m in range(1, M+1):
    for n in range(m, NQ+1):
        for l in range(0, m+1):
            D = 4*m*n - l*l
            if not (3*m*m <= D < 200): continue
            d = dstar(out['1A'], m, n, l); d2 = dstar(out['2A'], m, n, l); d3 = dstar(out['3A'], m, n, l); d4 = dstar(out['4B'], m, n, l)
            S2 = [Fr(d + d2, 2), Fr(d - d2, 2)]
            S3 = [Fr(d + 2*d3, 3), Fr(d - d3, 3)]
            S4 = [Fr(d + 2*d4 + d2, 4), Fr(d - d2, 4), Fr(d - 2*d4 + d2, 4)]
            for tag, S in (('Z2', S2), ('Z3', S3), ('Z4', S4)):
                for s in S:
                    if s <= 0 or s.denominator != 1: bad.append((tag, m, n, l, s))
            minratio[2] = min(minratio[2], d/abs(d2) if d2 else 1e9)
            minratio[3] = min(minratio[3], d/(2*abs(d3)) if d3 else 1e9)
            minratio[4] = min(minratio[4], d/(2*abs(d4) + abs(d2)))
            cnt += 1
print('charges checked', cnt, 'failures', bad[:10], len(bad))
print('min d*/|d_2A| %.3f   min d*/(2|d_3A|) %.3f   min d*/(2|d_4B|+|d_2A|) %.3f' % (minratio[2], minratio[3], minratio[4]))
