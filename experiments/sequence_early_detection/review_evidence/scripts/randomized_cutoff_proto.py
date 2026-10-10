"""Prototype (scratch, synthetic scores only): a validation-fitted RANDOMISED cut-off that hits a target FPR despite ties.
Rule fitted on validation only:  alert if score > t ; if score == t alert with probability q.  (t, q) chosen so expected val FPR == target.
Applied to test with a per-row pseudo-random draw derived from a hash of the transaction id (reproducible, label-free)."""
import hashlib, numpy as np
def fit_cutoff(y, s, target):
    neg = np.sort(s[y == 0])[::-1]; n = len(neg); allowed = target * n
    k = int(np.floor(allowed))                    # whole false alarms allowed
    t = neg[k] if k < n else neg[-1]              # (k+1)-th largest negative score
    above = int((neg > t).sum()); tied = int((neg == t).sum())
    q = float(np.clip((allowed - above) / tied, 0.0, 1.0)) if tied else 0.0
    return float(t), q
def draw(ids):  # uniform(0,1) from the id only
    return np.array([int(hashlib.sha256(i.encode()).hexdigest()[:12], 16) / 16**12 for i in ids])
def alerts(s, ids, t, q): return (s > t) | ((s == t) & (draw(ids) < q))
rng = np.random.default_rng(1)
def rf_like(n_trees, n_neg, n_pos, noise):
    neg = rng.binomial(n_trees, np.where(rng.random(n_neg) < noise, 0.02, 0.0005)) / n_trees
    pos = rng.binomial(n_trees, rng.beta(1.2, 2.5, n_pos)) / n_trees
    return np.r_[neg, pos], np.r_[np.zeros(n_neg), np.ones(n_pos)]
print("target | val FPR (rule fitted here) | independent TEST sample FPR | test recall   (RF-like tie-heavy scores, 100 trees, 25% noisy negatives)")
sv, yv = rf_like(100, 240_000, 1_000, .25); st, yt = rf_like(100, 240_000, 1_000, .25)
idv = [f"v{i}" for i in range(len(sv))]; idt = [f"t{i}" for i in range(len(st))]
for target in (0.01, 0.001):
    t, q = fit_cutoff(yv, sv, target)
    a_v = alerts(sv, idv, t, q); a_t = alerts(st, idt, t, q)
    print(f"{target:>6g} | {a_v[yv==0].mean():.5f}                   | {a_t[yt==0].mean():.5f}                    | {a_t[yt==1].mean():.3f}   (t={t:.4f}, q={q:.3f})")
