"""Online gradient-geometry metrics for f=0 baselines.

Computed on the honest, post-momentum vectors actually collected for
aggregation -- no raw vectors ever leave this module, only scalar/small-array
summaries. Everything stays vectorized on the input tensors' own device.
"""
import torch


def compute_layer_boundaries(model):
    """Groups model.named_parameters() into contiguous (name, start, end)
    index ranges in the flattened parameter vector, one range per submodule
    (weight+bias of the same layer collapse into one block, since they're
    adjacent in declaration order)."""
    model = getattr(model, "module", model)  # unwrap DataParallel if present
    boundaries = []
    offset = 0
    current_name, current_start = None, 0
    for name, param in model.named_parameters():
        n = param.numel()
        layer_name = name.rsplit(".", 1)[0] if "." in name else name
        if layer_name != current_name:
            if current_name is not None:
                boundaries.append((current_name, current_start, offset))
            current_name, current_start = layer_name, offset
        offset += n
    if current_name is not None:
        boundaries.append((current_name, current_start, offset))
    return boundaries


def _sign_agreement(stacked, eps=1e-12):
    """A_j per coordinate: fraction of honest clients agreeing with the
    majority sign at that coordinate. |m_j| < eps abstains (counts toward
    neither sign; n stays fixed in the denominator)."""
    n = stacked.shape[0]
    pos = (stacked > eps).sum(dim=0)
    neg = (stacked < -eps).sum(dim=0)
    return torch.maximum(pos, neg).float() / n


def compute_geometry_metrics(honest_vectors, layer_boundaries, agreement_threshold=0.8, hist_bins=20):
    """
    honest_vectors: list of 1-D tensors, post-momentum, honest clients only.
    layer_boundaries: output of compute_layer_boundaries().
    Returns a flat dict of scalars, ready to become one CSV row.
    """
    stacked = torch.stack(honest_vectors, dim=0)  # (n, d)
    n = stacked.shape[0]

    mu = stacked.mean(dim=0)
    sigma = stacked.std(dim=0, unbiased=False)
    mu_sq = mu.pow(2)

    norms = stacked.norm(dim=1)  # per-client vector norms
    dispersion_n = (norms.std(unbiased=False) / (norms.mean() + 1e-12)).item()

    agreement = _sign_agreement(stacked)  # A_j, one per coordinate
    consensus_mask = agreement >= agreement_threshold
    consensus_q = consensus_mask.float().mean().item()
    consensus_s = ((mu_sq * consensus_mask).sum() / (mu_sq.sum() + 1e-12)).item()

    normed = stacked / (norms.unsqueeze(1) + 1e-12)
    cos_matrix = normed @ normed.T
    iu = torch.triu_indices(n, n, offset=1, device=stacked.device)
    pair_cos = cos_matrix[iu[0], iu[1]]

    row = {
        "dispersion_N": dispersion_n,
        "consensus_Q": consensus_q,
        "consensus_S": consensus_s,
        "cos_mean": pair_cos.mean().item(),
        "cos_min": pair_cos.min().item(),
        "cos_max": pair_cos.max().item(),
        "cv_median": (sigma / (mu.abs() + 1e-12)).median().item(),
    }
    for i, norm_val in enumerate(norms.tolist()):
        row[f"norm_{i}"] = norm_val

    for layer_name, start, end in layer_boundaries:
        layer_mu_sq = mu_sq[start:end]
        layer_mask = consensus_mask[start:end]
        if layer_mask.numel() == 0:
            continue
        row[f"Q_{layer_name}"] = layer_mask.float().mean().item()
        row[f"S_{layer_name}"] = ((layer_mu_sq * layer_mask).sum() / (layer_mu_sq.sum() + 1e-12)).item()
        abs_mu_layer = mu[start:end].abs()
        q25, q50, q75 = torch.quantile(
            abs_mu_layer, torch.tensor([0.25, 0.5, 0.75], device=abs_mu_layer.device)
        ).tolist()
        row[f"absmu_q25_{layer_name}"] = q25
        row[f"absmu_median_{layer_name}"] = q50
        row[f"absmu_q75_{layer_name}"] = q75

    # Replayable histograms of A_j over [0.5, 1.0] -- lets other consensus
    # thresholds be tested later without rerunning training.
    hist = torch.histc(agreement, bins=hist_bins, min=0.5, max=1.0)
    hist = hist / hist.sum().clamp(min=1)
    for i, v in enumerate(hist.tolist()):
        row[f"Ahist_{i:02d}"] = v

    bin_width = 0.5 / hist_bins
    bin_idx = ((agreement - 0.5) / bin_width).clamp(0, hist_bins - 1 - 1e-9).long()
    weighted_hist = torch.zeros(hist_bins, device=stacked.device)
    weighted_hist.scatter_add_(0, bin_idx, mu_sq)
    weighted_hist = weighted_hist / weighted_hist.sum().clamp(min=1e-12)
    for i, v in enumerate(weighted_hist.tolist()):
        row[f"AhistW_{i:02d}"] = v

    return row
