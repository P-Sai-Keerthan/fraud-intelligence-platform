"""
Monte-Carlo check of the planned primary analysis. NO model output, NO dataset: purely simulated paired hit/miss
outcomes for two detectors on first-fraud events that fall in clusters (rings, repeat victims, households).

Questions answered:
  1. Does the paired CLUSTER bootstrap give ~95 % coverage for the pooled recall difference? (and the naive event-level one?)
  2. What is the power of the v0 decision rule (LB>0 and point>=0.05) versus a three-way classification, as a function of the true difference?
  3. How does the design effect from clustering change the needed sample size?
"""
import json, sys
import numpy as np
from scipy.stats import norm

RNG = np.random.default_rng(20261010)
EVENT_SHARE = {1: .65, 2: .17, 3: .065, 4: .04, 5: .04, 6: .035}     # share of EVENTS in clusters of this size (assumed)
SIZES = np.array(list(EVENT_SHARE)); P_SIZE = np.array([EVENT_SHARE[s] / s for s in SIZES]); P_SIZE /= P_SIZE.sum()


def make_clusters(n_events, rng):
    sizes = []
    while sum(sizes) < n_events:
        sizes.append(int(rng.choice(SIZES, p=P_SIZE)))
    sizes[-1] -= sum(sizes) - n_events
    return np.array([s for s in sizes if s > 0])


def simulate_dataset(sizes, rB, delta, rho_c, r_ab, rng):
    """Gaussian-copula paired outcomes. Shared cluster effect (rho_c) acts on BOTH detectors; r_ab = event-level latent correlation."""
    k = len(sizes)
    u = rng.standard_normal(k)
    cl = np.repeat(np.arange(k), sizes)
    n = len(cl)
    wa = rng.standard_normal(n)
    wb = r_ab * wa + np.sqrt(1 - r_ab ** 2) * rng.standard_normal(n)
    za = np.sqrt(rho_c) * u[cl] + np.sqrt(1 - rho_c) * wa
    zb = np.sqrt(rho_c) * u[cl] + np.sqrt(1 - rho_c) * wb
    a = za <= norm.ppf(rB + delta)
    b = zb <= norm.ppf(rB)
    sa = np.bincount(cl, weights=a, minlength=k); sb = np.bincount(cl, weights=b, minlength=k)
    return sa - sb, sizes.astype(float), a, b


def boot_ci(diff_by_ds, size_by_ds, reps, rng):
    """Stratified (by dataset) cluster bootstrap of the pooled, event-weighted difference. Returns (point, lo, hi)."""
    num = np.zeros(reps); den = np.zeros(reps)
    for d, s in zip(diff_by_ds, size_by_ds):
        k = len(d)
        counts = rng.multinomial(k, np.full(k, 1.0 / k), size=reps)
        num += counts @ d; den += counts @ s
    est = np.array(sum(d.sum() for d in diff_by_ds)) / sum(s.sum() for s in size_by_ds)
    dist = num / den
    return est, *np.quantile(dist, [0.025, 0.975])


def calibrate_rab(rB, rho_c, target_d, rng, n=400_000):
    """latent correlation r_ab giving the wanted discordance d = P(A != B) (at delta = 0.05)."""
    best = None
    for r in np.linspace(0.2, 0.98, 40):
        sizes = np.full(n // 1, 1)
        _, _, a, b = simulate_dataset(sizes, rB, 0.05, rho_c, r, rng)
        d = np.mean(a != b)
        if best is None or abs(d - target_d) < abs(best[1] - target_d):
            best = (float(r), float(d))
    return best


def run(n_ds, n_ev, delta, rho_c, r_ab, sims, reps, rng, rB=0.55, margin=0.05):
    out = dict(cover_cluster=0, cover_naive=0, w_cluster=0.0, w_naive=0.0, lb_pos=0, v0_adv=0, no_meaningful=0, point_ge=0,
               point_sd=[], disc=[])
    truth = delta        # marginal difference rA - rB is exactly delta by construction
    for _ in range(sims):
        dd, ss, naive_d, naive_s = [], [], [], []
        for _ in range(n_ds):
            sizes = make_clusters(n_ev, rng)
            d, s, a, b = simulate_dataset(sizes, rB, delta, rho_c, r_ab, rng)
            dd.append(d); ss.append(s)
            naive_d.append((a.astype(float) - b.astype(float))); naive_s.append(np.ones(len(a)))
            out["disc"].append(np.mean(a != b))
        est, lo, hi = boot_ci(dd, ss, reps, rng)
        _, nlo, nhi = boot_ci(naive_d, naive_s, reps, rng)
        out["cover_cluster"] += lo <= truth <= hi; out["cover_naive"] += nlo <= truth <= nhi
        out["w_cluster"] += hi - lo; out["w_naive"] += nhi - nlo
        out["lb_pos"] += lo > 0
        out["v0_adv"] += (lo > 0) and (est >= margin)
        out["no_meaningful"] += hi < margin
        out["point_ge"] += est >= margin
        out["point_sd"].append(est)
    return dict(n_datasets=n_ds, events=n_ds * n_ev, true_delta=delta, sims=sims,
                coverage_cluster=out["cover_cluster"] / sims, coverage_naive=out["cover_naive"] / sims,
                mean_width_cluster=round(out["w_cluster"] / sims, 4), mean_width_naive=round(out["w_naive"] / sims, 4),
                P_LB_gt_0=out["lb_pos"] / sims, P_v0_rule_advantage=out["v0_adv"] / sims, P_point_ge_margin=out["point_ge"] / sims,
                P_UB_lt_margin=out["no_meaningful"] / sims, sd_of_point_estimate=round(float(np.std(out["point_sd"])), 4),
                mean_discordance=round(float(np.mean(out["disc"])), 3))


if __name__ == "__main__":
    SIMS, REPS = int(sys.argv[1]), int(sys.argv[2])
    rho_c = 0.30
    r_ab, d_obs = calibrate_rab(0.55, rho_c, 0.25, RNG)
    print(json.dumps({"assumed_cluster_event_shares": EVENT_SHARE, "latent_within_cluster_corr": rho_c,
                      "calibrated_event_latent_corr": round(r_ab, 3), "implied_discordance_at_delta_0.05": round(d_obs, 3)}), flush=True)
    rows = []
    for n_ds, delta in [(5, 0.0), (5, 0.03), (5, 0.05), (5, 0.08), (5, 0.10), (3, 0.05), (3, 0.08), (1, 0.05), (1, 0.10)]:
        res = run(n_ds, 440, delta, rho_c, r_ab, SIMS, REPS, RNG)
        rows.append(res); print(json.dumps(res), flush=True)
