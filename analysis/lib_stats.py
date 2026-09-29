import numpy as np
from itertools import combinations


def majority_vote(labels):

    import collections
    clean = [l for l in labels
             if l is not None and not (isinstance(l, float) and np.isnan(l))]
    if not clean:
        return (None, 0, 0)
    counts = collections.Counter(clean)
    top = max(counts.values())
    winners = [k for k, v in counts.items() if v == top]
    winner = winners[0] if len(winners) == 1 else "unclear"
    return (winner, int(top), int(len(clean)))


def fleiss_kappa(rating_counts):

    m = np.asarray(rating_counts, dtype=float)
    N, C = m.shape
    n = m.sum(axis=1)
    if not np.allclose(n, n[0]):
        # unequal rater counts per item: drop items missing a rater is caller's job;
        # use the per-item n in the standard generalization
        pass
    n_bar = n.mean()
    p_j = m.sum(axis=0) / m.sum()                      # overall share per category
    P_i = (np.square(m).sum(axis=1) - n) / (n * (n - 1))  # agreement within each item
    P_bar = np.nanmean(P_i)
    P_e = np.square(p_j).sum()                          # chance agreement
    if np.isclose(P_e, 1.0):
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def krippendorff_alpha(units, level="nominal"):

    units = [np.asarray([v for v in u if v is not None and not
                         (isinstance(v, float) and np.isnan(v))], dtype=float) for u in units]
    units = [u for u in units if len(u) >= 2]
    values = np.unique(np.concatenate(units))
    index = {v: i for i, v in enumerate(values)}
    # counts[u, c] = how many judges gave unit u the value c
    counts = np.zeros((len(units), len(values)))
    for row, u in enumerate(units):
        for v in u:
            counts[row, index[v]] += 1
    m = counts.sum(axis=1)
    # coincidence matrix: every ordered pair of ratings within a unit, weighted 1/(m_u - 1)
    weighted = counts / (m - 1)[:, None]
    coincidence = counts.T @ weighted - np.diag(weighted.sum(axis=0))
    n_c = coincidence.sum(axis=0)
    n = n_c.sum()
    if level == "nominal":
        delta = 1.0 - np.eye(len(values))
    elif level == "interval":
        delta = np.subtract.outer(values, values) ** 2
    elif level == "ordinal":
        cum = np.cumsum(n_c)
        lo, hi = np.minimum.outer(np.arange(len(values)), np.arange(len(values))), \
                 np.maximum.outer(np.arange(len(values)), np.arange(len(values)))
        between = cum[hi] - np.where(lo > 0, cum[lo - 1], 0.0)   # sum of n_g for g in [c, k]
        delta = (between - (n_c[lo] + n_c[hi]) / 2) ** 2
    else:
        raise ValueError(f"unknown level: {level}")
    observed = (coincidence * delta).sum()
    expected = (np.outer(n_c, n_c) * delta).sum() / (n - 1)
    return float(1.0 - observed / expected) if expected > 0 else 1.0


def wilson_ci(k, n, z=1.96):

    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / d
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / d
    return (p, max(0.0, center - half), min(1.0, center + half))


def cluster_bootstrap(df, cluster_col, stat_fn, n_boot=2000, seed=0):

    rng = np.random.default_rng(seed)
    clusters = df[cluster_col].to_numpy()
    uniq = np.unique(clusters)
    # pre-index rows per cluster for speed
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    est = float(stat_fn(df))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_cluster[c] for c in pick])
        boots[b] = stat_fn(df.iloc[rows])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return (est, float(lo), float(hi))


def mcnemar_exact(b, c):

    from math import comb
    n = b + c
    if n == 0:
        return (0, 1.0, "none")
    k = min(b, c)
    # two-sided exact binomial p at prob 0.5
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p = min(1.0, 2 * tail)
    direction = "b>c" if b > c else ("c>b" if c > b else "equal")
    return (n, float(p), direction)


def paired_counts(df, key_cols, cond_col, cond_a, cond_b, outcome_col, good_value):

    a = df[df[cond_col] == cond_a].set_index(key_cols)[outcome_col]
    b = df[df[cond_col] == cond_b].set_index(key_cols)[outcome_col]
    j = a.to_frame("a").join(b.to_frame("b"), how="inner").dropna()
    ag = j["a"] == good_value
    bg = j["b"] == good_value
    both_good = int((ag & bg).sum())
    both_bad = int((~ag & ~bg).sum())
    a_good_b_bad = int((ag & ~bg).sum())
    a_bad_b_good = int((~ag & bg).sum())
    return (len(j), both_good, both_bad, a_good_b_bad, a_bad_b_good)
