"""Online gradient-structure metrics for the f=0 SNN-vs-CNN gradient study.

Same discipline as gradient_geometry.py: computed directly on the honest,
post-momentum vectors already in memory during training -- no raw vectors
ever leave this module or hit disk, only scalar/small-array summaries (a
prior raw-vector-dump approach exhausted disk on a full sweep; this module
exists so that mistake isn't repeated for this experiment).

Per-client norms and honest-honest cosine similarity are already covered by
gradient_geometry.compute_geometry_metrics (norm_i / cos_mean / cos_min /
cos_max) -- this module adds the metrics that one doesn't: PCA effective
rank, active-coordinate magnitude, support overlap, and a fixed-bin
coordinate-value histogram for a small fixed client subset.
"""
import torch


def pick_fixed_subset(nb_honest_clients, subset_size=3, seed=0):
    """One fixed subset of client indices, chosen once per run (not
    resampled per step) so the same clients are compared across training."""
    generator = torch.Generator().manual_seed(seed)
    k = min(subset_size, nb_honest_clients)
    return torch.randperm(nb_honest_clients, generator=generator)[:k].tolist()


def compute_gradient_structure_metrics(
    honest_vectors,
    subset_idx,
    active_thresholds=(0.0, 1e-6, 1e-5, 1e-4, 1e-3),
    topk_support=100,
    pca_components=3,
    hist_bins=60,
):
    """
    honest_vectors: list of 1-D tensors, post-momentum, honest clients only.
    subset_idx: fixed list of client indices (see pick_fixed_subset), used
        for the per-client coordinate-value histogram.
    Returns a flat dict of scalars, ready to become one CSV row.
    """
    stacked = torch.stack(honest_vectors, dim=0)  # (n, d)
    n, d = stacked.shape
    row = {}

    # --- PCA effective rank (mean-centered honest vectors) ---
    centered = stacked - stacked.mean(dim=0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    eigvals = s.pow(2)
    total = eigvals.sum()
    if total > 1e-18:
        ratio = eigvals / total
        cum = torch.cumsum(ratio, dim=0)
        n90 = int((cum < 0.9).sum().item()) + 1
        participation_ratio = (eigvals.sum() ** 2 / (eigvals.pow(2).sum() + 1e-18)).item()
    else:
        n90, participation_ratio = 0, 0.0
    row["pca_n_components_90pct"] = n90
    row["pca_participation_ratio"] = participation_ratio

    k = min(pca_components, vh.shape[0])
    projected = centered @ vh[:k].T  # (n, k)
    for i in range(n):
        for j in range(k):
            row[f"pca_proj_c{i}_pc{j}"] = projected[i, j].item()
    if n > 1:
        pdists = torch.pdist(projected)
        row["pca_proj_dist_mean"] = pdists.mean().item()
        row["pca_proj_dist_max"] = pdists.max().item()
    else:
        row["pca_proj_dist_mean"] = 0.0
        row["pca_proj_dist_max"] = 0.0

    # --- active-coordinate mean magnitude, swept over thresholds ---
    abs_stacked = stacked.abs()
    for th in active_thresholds:
        mask = abs_stacked >= th
        client_means = []
        for i in range(n):
            m = mask[i]
            client_means.append(abs_stacked[i][m].mean().item() if m.any() else float("nan"))
        means_t = torch.tensor(client_means)
        valid = means_t[~torch.isnan(means_t)]
        row[f"active_mean_thr_{th:g}"] = valid.mean().item() if len(valid) else float("nan")
        row[f"active_frac_thr_{th:g}"] = mask.float().mean().item()

    # --- support overlap: Jaccard of top-k coordinate sets (by |value|) ---
    k_support = min(topk_support, d)
    topk_idx = torch.topk(abs_stacked, k_support, dim=1).indices  # (n, k_support)
    supports = [set(idx.tolist()) for idx in topk_idx]
    jaccards = []
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(supports[i] & supports[j])
            union = len(supports[i] | supports[j])
            jaccards.append(inter / union if union else 0.0)
    if jaccards:
        mean_j = sum(jaccards) / len(jaccards)
        row["jaccard_topk_mean"] = mean_j
        row["jaccard_topk_std"] = (sum((x - mean_j) ** 2 for x in jaccards) / len(jaccards)) ** 0.5
    else:
        row["jaccard_topk_mean"] = 0.0
        row["jaccard_topk_std"] = 0.0

    # --- fixed-subset coordinate-value histogram (bin counts only, range
    # scaled to this step's own magnitude -- never the raw vectors) ---
    scale = abs_stacked.max().item()
    lo, hi = -scale, scale
    if hi <= lo:
        hi = lo + 1e-12
    for pos, ci in enumerate(subset_idx):
        hist = torch.histc(stacked[ci], bins=hist_bins, min=lo, max=hi)
        for b, v in enumerate(hist.tolist()):
            row[f"subset{pos}_client{ci}_hist_{b:03d}"] = v
    row["subset_hist_lo"] = lo
    row["subset_hist_hi"] = hi
    row["subset_hist_bins"] = hist_bins

    return row
