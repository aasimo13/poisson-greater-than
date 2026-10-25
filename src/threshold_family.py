"""Sum thresholds are comparisons on a smaller alphabet.

Two senders hold A, B in {0..K-1}. Margin d means f = [A - B >= d], with integer 1 <= d <= K-1.
1. [A > B] == [A + (K-1-B) >= K], and [A+B >= T] == not[(K-1-A)+(K-1-B) >= 2K-1-T],
   so every nonconstant integer sum threshold is a margin problem up to relabeling and flipping the output.
   Its equivalent comparator has L = min(T+1, 2K-T) levels for 1 <= T <= 2K-2;
   T <= 0 and T >= 2K-1 are constant predicates with zero signal cost.
2. Margin d on K levels == A > B on K-d+1 levels, in both directions:
   merging levels A' = max(A-d+1, 0), B' = min(B, K-d) maps a K-d+1 design onto a
   margin d design at the same cost, and restricting to A = A'+d-1 maps back.
   So the optimal photon budgets are equal (e.g. margin 2 on K=4 is the K=3 comparator).
3. High-precision check of a deterministic sum threshold [A+B >= 2] at 50 digits:
   x = y = [4.6054, 0, 29.3097, 29.3097], decoder "true iff N == 0 or N >= 18".
   It is equivalent to margin 2 for K=4. To implement [A-B >= 2] directly, reverse
   A's encoder to x[3-A], keep B's encoder, and complement the decoder.
"""
import mpmath as mp

def equivalences(kmax=8):
    for K in range(2, kmax + 1):
        r = range(K)
        assert all((a > b) == (a + (K - 1 - b) >= K) for a in r for b in r)
        for T in range(0, 2 * K):
            assert all((a + b >= T) == (not ((K - 1 - a) + (K - 1 - b) >= 2 * K - 1 - T)) for a in r for b in r)
        for d in range(1, K):
            assert all((a - b >= d) == (max(a - d + 1, 0) > min(b, K - d)) for a in r for b in r)
            assert all(((a + d - 1) - b >= d) == (a > b) for a in range(K - d + 1) for b in range(K - d + 1))
    return kmax

def margin2_design():
    """Check the equivalent sum predicate A+B >= 2; the legacy function name is retained."""
    mp.mp.dps = 50
    x = [mp.mpf("4.6054"), mp.mpf(0), mp.mpf("29.3097"), mp.mpf("29.3097")]
    pmf = lambda n, l: mp.e ** (-l) * l ** n / mp.factorial(n)
    worst, at = mp.mpf(0), None
    for a in range(4):
        for b in range(4):
            lam = x[a] + x[b]
            p_true = pmf(0, lam) + 1 - mp.fsum(pmf(n, lam) for n in range(18))
            err = 1 - p_true if a + b >= 2 else p_true
            if err > worst:
                worst, at = err, (a, b)
    return max(x) * 2, worst, at

if __name__ == "__main__":
    k = equivalences()
    print(f"Equivalences hold exhaustively for K = 2..{k}")
    m, w, at = margin2_design()
    print(f"Relabeled margin 2 design (A+B >= 2): max mean = {mp.nstr(m, 8)}, worst error = {mp.nstr(100 * w, 8)}% at (A,B)={at}")
