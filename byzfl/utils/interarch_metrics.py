"""Inter-architecture gradient geometry metrics for f=0 baseline experiments.

Computes dispersion, consensus, participation and scale metrics on the
honest client vectors (pre- or post-momentum) at each training step.
Everything stays vectorised on the input tensors' device (GPU).

Metrics implemented (spec numbering):
  (2)  N            — CV of per-client vector norms
  (2b) norm_k       — individual client norms
  (3)  CV_eff       — effective CV: sqrt(sum sigma_j^2) / sqrt(sum mu_j^2)

Note: all variances use unbiased=False (divisor n, not n-1), conforming to
the spec formula (1/n)Σ. Do NOT change to unbiased=True — it would break
comparability with runs already completed under this convention.
  (4)  scale, σ_H   — mean norm, sqrt(sum sigma_j^2)
  (5)  cos_mean/min/max — pairwise cosine similarities
  (6)  Pi           — signal-weighted participation ratio
  (7)  CV_eff_act   — CV_eff restricted to active clients per coordinate
  (8)  per-layer    — CV_eff(L), Pi(L), median|mu_j|(L)
  (9)  firing rates — passed through from hooks (SNN only)
"""

import torch


def compute_interarch_metrics(vectors, layer_boundaries, firing_rates, prefix=""):
    """Compute all inter-architecture metrics for one set of client vectors.

    Parameters
    ----------
    vectors : list[Tensor]
        List of n 1-D tensors (flat gradient vectors), one per honest client.
    layer_boundaries : list[tuple(str, int, int)]
        Output of ``compute_layer_boundaries()``: (layer_name, start, end).
    firing_rates : dict[str, float]
        Mean firing rate per layer name (SNN only). Empty dict for CNN.
    prefix : str
        Column prefix: ``""`` for post-momentum, ``"g_"`` for pre-momentum.

    Returns
    -------
    dict[str, float]
        Flat dict of scalar metrics, ready to become one CSV row (or merged
        with another call's output).
    """
    stacked = torch.stack(vectors, dim=0)  # (n, d)
    n, d = stacked.shape

    # ── (1) Per-coordinate mean and variance ──────────────────────────────
    mu = stacked.mean(dim=0)                           # (d,)
    sigma_sq = stacked.var(dim=0, unbiased=False)      # (d,)  — divisor n, see module docstring

    # ── (2) N — dispersion of norms ───────────────────────────────────────
    norms = stacked.norm(dim=1)                        # (n,)
    norm_mean = norms.mean()
    dispersion_N = (norms.std(unbiased=False) / (norm_mean + 1e-12)).item()

    # ── (3) CV_eff ────────────────────────────────────────────────────────
    sum_sigma_sq = sigma_sq.sum()
    mu_sq = mu.pow(2)
    sum_mu_sq = mu_sq.sum()
    cv_eff = (sum_sigma_sq.sqrt() / (sum_mu_sq + 1e-24).sqrt()).item()

    # ── (4) Scale and sigma_H ─────────────────────────────────────────────
    scale = norm_mean.item()
    sigma_H = sum_sigma_sq.sqrt().item()

    # ── (5) Pairwise cosine similarities ──────────────────────────────────
    normed = stacked / (norms.unsqueeze(1) + 1e-12)
    cos_matrix = normed @ normed.T
    iu = torch.triu_indices(n, n, offset=1, device=stacked.device)
    pair_cos = cos_matrix[iu[0], iu[1]]
    cos_mean = pair_cos.mean().item()
    cos_min = pair_cos.min().item()
    cos_max = pair_cos.max().item()

    # ── (6) Participation π_j and Π (signal-weighted) ─────────────────────
    # eps_L relative per layer: 1e-3 * median of NON-ZERO |m_j^(k)| over
    # that layer.  Using the full median (zeros included) would make eps_L ≈ 0
    # on ReLU layers where >50% of entries are near-zero, collapsing Π → 1
    # and blinding the participation metric on the architecture it was built for.
    active = torch.ones_like(stacked, dtype=torch.bool)  # (n, d)
    for _layer_name, start, end in layer_boundaries:
        layer_abs = stacked[:, start:end].abs()
        nonzero = layer_abs[layer_abs > 0]
        if nonzero.numel() > 0:
            eps_L = 1e-3 * nonzero.median()
        else:
            eps_L = torch.tensor(0.0, device=stacked.device)
        active[:, start:end] = layer_abs > eps_L

    active_float = active.float()                      # (n, d)
    pi_j = active_float.mean(dim=0)                    # (d,) fraction of clients active at coord j

    # Signal-weighted participation
    Pi_num = (mu_sq * pi_j).sum()
    Pi = (Pi_num / (sum_mu_sq + 1e-24)).item()

    # ── (7) CV_eff_act — CV_eff on active clients only ────────────────────
    # For each coordinate j, compute mu_tilde and sigma_tilde only over
    # clients where a_j^(k) = 1. If <= 1 client active, sigma_tilde = 0.
    #
    # Computed in float64 to avoid catastrophic cancellation in
    # E[X²] − E[X]² when |mu_tilde| >> sigma_tilde (common for gradients).
    count_active = active_float.sum(dim=0)             # (d,)

    stacked_f64 = stacked.double()
    active_f64 = active_float.double()

    masked_f64 = stacked_f64 * active_f64              # zero out inactive
    sum_active = masked_f64.sum(dim=0)                 # (d,)
    safe_count = count_active.double().clamp(min=1)
    mu_tilde = sum_active / safe_count                 # (d,)

    sum_sq_active = (stacked_f64.pow(2) * active_f64).sum(dim=0)
    var_tilde = sum_sq_active / safe_count - mu_tilde.pow(2)
    var_tilde = var_tilde.clamp(min=0)                 # numerical safety
    # Zero out variance where <= 1 client active
    var_tilde = torch.where(count_active.double() > 1, var_tilde, torch.zeros_like(var_tilde))

    sum_var_tilde = var_tilde.sum()
    sum_mu_tilde_sq = mu_tilde.pow(2).sum()
    cv_eff_act = (sum_var_tilde.sqrt() / (sum_mu_tilde_sq + 1e-24).sqrt()).item()

    # ── Build output row ──────────────────────────────────────────────────
    p = prefix
    row = {
        f"{p}N": dispersion_N,
        f"{p}CV_eff": cv_eff,
        f"{p}scale": scale,
        f"{p}sigma_H": sigma_H,
        f"{p}cos_mean": cos_mean,
        f"{p}cos_min": cos_min,
        f"{p}cos_max": cos_max,
        f"{p}Pi": Pi,
        f"{p}CV_eff_act": cv_eff_act,
    }

    # Individual norms
    for i, norm_val in enumerate(norms.tolist()):
        row[f"{p}norm_{i}"] = norm_val

    # ── (8) Per-layer metrics ─────────────────────────────────────────────
    for layer_name, start, end in layer_boundaries:
        layer_mu_sq = mu_sq[start:end]
        layer_sigma_sq = sigma_sq[start:end]
        layer_sum_mu_sq = layer_mu_sq.sum()
        layer_sum_sigma_sq = layer_sigma_sq.sum()

        # CV_eff per layer
        layer_cv_eff = (layer_sum_sigma_sq.sqrt() / (layer_sum_mu_sq + 1e-24).sqrt()).item()
        row[f"{p}CV_eff_{layer_name}"] = layer_cv_eff

        # Pi per layer
        layer_pi_j = pi_j[start:end]
        layer_Pi = ((layer_mu_sq * layer_pi_j).sum() / (layer_sum_mu_sq + 1e-24)).item()
        row[f"{p}Pi_{layer_name}"] = layer_Pi

        # Median |mu_j| per layer
        layer_abs_mu = mu[start:end].abs()
        row[f"{p}absmu_med_{layer_name}"] = layer_abs_mu.median().item()

    # ── (9) Firing rates (pass-through, no prefix duplication) ────────────
    if prefix == "" and firing_rates:
        for layer_name, rate in firing_rates.items():
            row[f"fr_{layer_name}"] = rate

    return row
