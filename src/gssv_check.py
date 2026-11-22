# Reproduces the published tables of Govindarajan-Samanta-Shanmugapriya-Virmani (arXiv:2205.08726, Tables 1-4):
# the values -B_6 = d*, -B_6^{g2}, -B_6^{g3}, -B_6^{g4} for (Q^2, P^2) up to (6,6) and Q.P = 0..4.
from psig import psi_coeffs, dstar
GSSV = {  # (Q^2,P^2): values at Q.P = 0,1,2,3,4 (None: outside the table)
 '1A': {(2, 2): [50064, 25353, 648, 327, 0], (2, 4): [1127472, 561576, 50064, 8376, -648], (4, 4): [32861184, 18458000, 3859456, 561576, 12800],
        (2, 6): [16491600, 8533821, 1127472, 130329, -15600], (4, 6): [632078672, 392427528, 110910300, 18458000, 1127472],
        (6, 6): [16193130552, 11232685725, 4173501828, 920577636, 110910300]},
 '2A': {(2, 2): [-1328, 505, -216, 55, 0], (2, 4): [-7696, 3128, -1328, 488, -104], (4, 4): [-44544, 19120, -7168, 3128, -1280],
        (2, 6): [-33520, 14781, -7696, 3209, -848], (4, 6): [-188528, 86232, -41316, 19120, -7696], (6, 6): [-723144, 353853, -243612, 126180, -41316]},
 '3A': {(2, 2): [-588, 297, -108, 30, 0], (2, 4): [-2442, 1272, -588, 204, -54], (4, 4): [-9696, 5390, -2696, 1272, -448],
        (2, 6): [-8964, 4998, -2442, 1026, -336], (4, 6): [-35446, 20256, -10956, 5390, -2442], (6, 6): [-127332, 76209, -42108, 22545, -10956]},
 '4B': {(2, 2): [-224, 117, -48, 19, 0], (2, 4): [-768, 444, -224, 84, -32], (4, 4): [-2624, 1608, -896, 444, -160],
        (2, 6): [-2400, 1437, -768, 373, -136], (4, 6): [-8128, 5148, -3028, 1608, -768], (6, 6): [-24712, 16117, -9828, 5652, -3028]},
}
if __name__ == '__main__':
    res = {'1A': psi_coeffs(4, 4, 1), '2A': psi_coeffs(4, 4, 2), '3A': psi_coeffs(4, 4, 3), '4B': psi_coeffs(4, 4, 4),
           '4B^2': psi_coeffs(4, 4, 4, twine=2)}
    ok = True; n = 0
    for g, tab in GSSV.items():
        for (Q2, P2), vals in tab.items():
            # the index is symmetric in Q^2 <-> P^2; FJ index m = Q^2/2 here, n = P^2/2
            m, nn = Q2//2, P2//2
            for l, v in enumerate(vals):
                mine = dstar(res[g], m, nn, l) if l <= 2*m else None
                mine2 = dstar(res[g], nn, m, l) if l <= 2*nn else None
                if mine is None and mine2 is None: continue
                good = (mine2 == v) and (mine is None or mine == v)
                n += 1
                if not good: ok = False; print('MISMATCH', g, Q2, P2, l, v, mine, mine2)
    # Z4: B^{g4^2} equals the 2A values
    for (Q2, P2), vals in GSSV['2A'].items():
        for l, v in enumerate(vals):
            if l <= Q2: assert dstar(res['4B^2'], P2//2, Q2//2, l) == v
    print('GSSV tables 1-4: %d entries compared, all equal: %s' % (n, ok))
