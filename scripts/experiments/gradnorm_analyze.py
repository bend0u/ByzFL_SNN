"""
Analyze + plot the gradnorm_probe.npz produced by gradnorm_probe.py.

Answers the design questions for a client-side raw-gradient clip that needs NO
SNN and NO online recalibration:
  1. Global raw grad-norm trajectory vs step, per gamma (does it drift? is a
     single fixed anchor defensible across heterogeneity?).
  2. Per-layer norm scale disparity (motivates per-layer vs global clip).
  3. AGC ratio ||g_layer|| / ||W_layer|| stability across step/layer/gamma
     (is the model-intrinsic, data-free anchor stable?).
  4. Per-neuron grad-norm distribution (motivates neuron-level clip).
Prints a percentile table and writes PNGs to results/activation_clip/gradnorm_probe/.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "activation_clip", "gradnorm_probe")
LAYERS = ["_c1", "_c2", "_f1", "_f2"]
GAMMAS = [1.0, 0.33, 0.0]
COL = {1.0: "#1f77b4", 0.33: "#ff7f0e", 0.0: "#d62728"}

d = np.load(os.path.join(OUT, "gradnorm_probe.npz"))


def pk(a, ps=(50, 80, 90, 95, 99, 100)):
    return np.percentile(a, ps)


# ---- percentile table (global raw grad norm) ----
print("\n=== RAW global grad norm percentiles (pooled over all steps x clients) ===")
print(f"{'gamma':>6} {'p50':>7} {'p80':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}")
for g in GAMMAS:
    v = d[f"g{g}_global"].ravel()
    p = pk(v)
    print(f"{g:>6} " + " ".join(f"{x:7.2f}" for x in p))
print("(reference: SNN-derived fixed clip = 21.0)")

# steady-state (last 200 steps) percentiles
print("\n=== RAW global grad norm percentiles (steps 300-499 only, steady state) ===")
print(f"{'gamma':>6} {'p50':>7} {'p80':>7} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>7}")
for g in GAMMAS:
    v = d[f"g{g}_global"][300:, :].ravel()
    print(f"{g:>6} " + " ".join(f"{x:7.2f}" for x in pk(v)))

# ---- plot 1: global norm trajectory (median + p10-p90 band across clients) ----
fig, ax = plt.subplots(figsize=(8, 4.5))
for g in GAMMAS:
    gn = d[f"g{g}_global"]  # [steps, clients]
    med = np.median(gn, axis=1)
    lo = np.percentile(gn, 10, axis=1)
    hi = np.percentile(gn, 90, axis=1)
    x = np.arange(len(med))
    ax.plot(x, med, color=COL[g], label=f"γ={g}")
    ax.fill_between(x, lo, hi, color=COL[g], alpha=0.15)
ax.axhline(21.0, color="k", ls="--", lw=1, label="SNN fixed clip = 21")
ax.set_xlabel("step"); ax.set_ylabel("raw grad L2 norm (per client)")
ax.set_title("Honest raw gradient norm vs step (median, p10–p90 band)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "global_norm_trajectory.png"), dpi=130)
plt.close(fig)

# ---- plot 2: per-layer norm trajectories (median across clients), gamma=0.33 ----
fig, ax = plt.subplots(figsize=(8, 4.5))
gref = 0.33
for L in LAYERS:
    ln = d[f"g{gref}_{L}"]
    ax.plot(np.median(ln, axis=1), label=L.lstrip("_"))
ax.set_xlabel("step"); ax.set_ylabel("raw grad L2 norm")
ax.set_title(f"Per-layer raw grad norm vs step (γ={gref}, median across clients)")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_layer_norm_trajectory.png"), dpi=130)
plt.close(fig)

# ---- plot 3: AGC ratio ||g||/||W|| per layer vs step (gamma=0.33) ----
fig, ax = plt.subplots(figsize=(8, 4.5))
print("\n=== AGC ratio ||g_layer||/||W_layer|| (steps 300-499, median[p10,p90]) ===")
for L in LAYERS:
    r = d[f"g{gref}_agc{L}"]
    ax.plot(np.median(r, axis=1), label=L.lstrip("_"))
    rs = r[300:, :].ravel()
    print(f"  {gref} {L}: {np.median(rs):.3f} [{np.percentile(rs,10):.3f},{np.percentile(rs,90):.3f}]")
ax.set_xlabel("step"); ax.set_ylabel("||g|| / ||W||")
ax.set_title(f"AGC ratio per layer vs step (γ={gref})")
ax.legend(); fig.tight_layout()
fig.savefig(os.path.join(OUT, "agc_ratio_trajectory.png"), dpi=130)
plt.close(fig)

# AGC ratio across gammas (steady state, per layer) -- is the data-free anchor gamma-stable?
print("\n=== AGC ratio steady-state median by (gamma, layer) ===")
print(f"{'layer':>5} " + " ".join(f"g={g:>4}" for g in GAMMAS))
for L in LAYERS:
    row = [np.median(d[f"g{g}_agc{L}"][300:, :]) for g in GAMMAS]
    print(f"{L:>5} " + " ".join(f"{x:7.3f}" for x in row))

# ---- plot 4: per-neuron grad-norm histograms per layer (gamma=0.33) ----
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for L, ax in zip(LAYERS, axes.ravel()):
    v = d[f"g{gref}_neuron{L}"]
    v = v[v > 0]
    ax.hist(np.log10(v + 1e-12), bins=60, color="#4c72b0")
    ax.set_title(f"{L.lstrip('_')}: per-neuron grad norm (log10), γ={gref}")
    ax.set_xlabel("log10 row-norm")
fig.suptitle("Per-neuron (output-unit) raw grad-norm distribution")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "per_neuron_hist.png"), dpi=130)
plt.close(fig)

print("\nsaved PNGs to", OUT)
