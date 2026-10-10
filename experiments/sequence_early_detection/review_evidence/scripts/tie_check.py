"""Do the existing cut-off functions behave on tie-heavy scores (random-forest-like)? Synthetic scores only; no data, no model."""
import ast, importlib.util, sys, numpy as np
SRC = sys.argv[1]
spec = importlib.util.spec_from_file_location("metrics_mod", f"{SRC}/backend/app/evaluation/metrics.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = open(f"{SRC}/backend/app/evaluation/multiseed.py").read(); tree = ast.parse(src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "tie_safe_cutoff")
ns = {"np": np}; exec(compile(ast.Module([fn], []), "x", "exec"), ns); tie_safe = ns["tie_safe_cutoff"]
rng = np.random.default_rng(0)
def rf_like(n_trees, n_neg, n_pos, leaf_noise):
    # RF probability = (# trees voting fraud)/n_trees -> discrete grid; most legit rows get 0 votes
    neg = rng.binomial(n_trees, np.where(rng.random(n_neg) < leaf_noise, 0.02, 0.0005)) / n_trees
    pos = rng.binomial(n_trees, rng.beta(1.2, 2.5, n_pos)) / n_trees
    return np.r_[neg, pos], np.r_[np.zeros(n_neg), np.ones(n_pos)]
print("target FPR | scores        | existing threshold_for_fpr realised val FPR | multiseed.tie_safe_cutoff realised val FPR")
for label, n_trees, leaf_noise in [("RF 300 trees, 8% noisy negs", 300, .08), ("RF 300 trees, 25% noisy negs", 300, .25), ("RF 100 trees, 25% noisy negs", 100, .25)]:
    s, y = rf_like(n_trees, 240_000, 1_000, leaf_noise)
    for target in (0.01, 0.001):
        t1 = m.threshold_for_fpr(y, s, target); fpr1 = float(((s >= t1) & (y == 0)).sum() / (y == 0).sum())
        t2 = tie_safe(y, s, target);           fpr2 = None if t2 is None else float(((s >= t2) & (y == 0)).sum() / (y == 0).sum())
        print(f"{target:>9g} | {label:<28} | {fpr1:.5f}                                  | {('none' if fpr2 is None else f'{fpr2:.5f}')}")
print("distinct score values in last example:", len(np.unique(s)))
