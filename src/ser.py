# 2D truncated series in q (0..NQ) and y (-Y..Y) with integer coefficients, Kronecker multiplication via FLINT
import numpy as np
from flint import fmpz_poly

class Ctx:
    def __init__(s, NQ, Y):
        s.NQ, s.Y = NQ, Y
        s.W = 4*Y + 1
    def zero(s):
        return np.zeros((s.NQ+1, 2*s.Y+1), dtype=object) * 0
    def enc(s, A):
        coeffs = [0]*((s.NQ+1)*s.W)
        nz = np.argwhere(A != 0)
        for n, jj in nz:
            coeffs[n*s.W + jj] = int(A[n, jj])
        return fmpz_poly(coeffs)
    def mul(s, A, B):
        P = s.enc(A) * s.enc(B)
        C = s.zero()
        cl = P.coeffs()
        W, Y, NQ = s.W, s.Y, s.NQ
        for e, c in enumerate(cl):
            if c == 0: continue
            n, r = divmod(e, W)
            if n > NQ: break
            j = r - 2*Y
            if -Y <= j <= Y:
                C[n, j+Y] = int(c)
        return C
    def mono(s, n, j, c=1):
        A = s.zero(); A[n, j+s.Y] = c; return A
