#!/bin/sh
# Runs every computation of the paper and writes the logs to logs/.
set -e
cd "$(dirname "$0")/src"
LOG=../logs
mkdir -p "$LOG"

# Twisted-twining genera and their projections (tables up to D = 20000, 2400 and 200).
for N in 2 3 4; do
  python3 genera.py $N 20000 > "$LOG/genera_N$N.log"
  python3 genera.py $N 2400 > /dev/null
  python3 genera.py $N 200 > /dev/null
done

# Tail bounds on boxes at every cusp: rho0 = 0.35, eta1 = 0.05, X1 = 30, theta = 0.04, s-ratio 1.02, 20 beta-bins.
for N in 2 3 4; do python3 tbox.py $N 0.35 0.05 30 0.04 1.02 20; done > "$LOG/tb_run.log"

# Constants of the lower bound for m <= 5.
python3 -c "from lowbd import lam_small; print([round(lam_small(m, 200), 4) for m in range(1, 6)])" > "$LOG/lam.log"

# Certificates for 1 <= m <= 2000 and the uniform lemma for m >= 101.
for N in 2 3 4; do python3 certN.py $N 1 2000 > "$LOG/certN${N}_1_2000.log"; done
for N in 2 3 4; do python3 largeN.py $N 101 > "$LOG/largeN_$N.log"; done

# Exact check of the small charges, and the two consistency checks.
python3 exact_small.py > "$LOG/exact_small.log"
for N in 2 3 4; do python3 cuspcheck.py $N > "$LOG/cuspcheck_$N.log"; done
python3 gssv_check.py > "$LOG/gssv_check.log"

echo "all computations finished; see $LOG"
