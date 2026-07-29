"""
Offline calibration of candidate client-side raw-grad clip mechanisms, using the
per-layer norms in gradnorm_probe.npz. Exact because the param vector partitions
into disjoint layer blocks, so ||g||^2 = sum_L ||g_L||^2.

Candidates:
  (A) global fixed cap C            -> clipped ||g|| = min(||g||, C)
  (B) AGC per-layer, threshold lam  -> ||g_L|| <- min(||g_L||, lam*||W_L||)
  (C) per-layer fixed caps c_L      -> ||g_L|| <- min(||g_L||, c_L)  (c_L = honest p99 per layer)

For each we report the resulting honest global-norm percentiles per gamma and the
fraction of (step,client) updates actually clipped -- we want a mechanism that,
like fixed-21, only shaves the tail (light clipping), NOT one that clips a large
fraction (which the adaptive-quantile clip did, and which failed).
"""
import os, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "results", "activation_clip", "gradnorm_probe")
LAYERS = ["_c1", "_c2", "_f1", "_f2"]
GAMMAS = [1.0, 0.33, 0.0]
d = np.load(os.path.join(OUT, "gradnorm_probe.npz"))

def gnorm(g):  # exact global norm from per-layer norms
    return np.sqrt(sum(d[f"g{g}_{L}"]**2 for L in LAYERS))  # [steps,clients]

print("=== baseline honest global-norm (unclipped) ===")
for g in GAMMAS:
    v = gnorm(g).ravel()
    print(f" g={g}: p90={np.percentile(v,90):.2f} p99={np.percentile(v,99):.2f} max={v.max():.2f}")

# (A) global fixed cap
print("\n=== (A) global fixed cap C: %updates clipped, per gamma ===")
for C in [15, 21, 25, 30]:
    row=[]
    for g in GAMMAS:
        v = gnorm(g).ravel()
        row.append(100*np.mean(v>C))
    print(f"  C={C:>3}: " + "  ".join(f"g{g}={r:5.1f}%" for g,r in zip(GAMMAS,row)))

# (B) AGC per-layer with single lambda
print("\n=== (B) AGC per-layer lambda: resulting honest global-norm p99 & %clipped(any layer) ===")
for lam in [0.02, 0.05, 0.08, 0.12, 0.2]:
    print(f"  lam={lam}:")
    for g in GAMMAS:
        clipped_layer = {L: np.minimum(d[f"g{g}_{L}"], lam*d[f"g{g}_w{L}"][:,None]) for L in LAYERS}
        gclip = np.sqrt(sum(clipped_layer[L]**2 for L in LAYERS))
        # any layer clipped?
        any_clip = np.zeros_like(d[f"g{g}__c1"], dtype=bool)
        for L in LAYERS:
            any_clip |= d[f"g{g}_{L}"] > lam*d[f"g{g}_w{L}"][:,None]
        base = gnorm(g)
        shrink = 100*(1 - gclip.mean()/base.mean())
        print(f"     g={g}: p99_after={np.percentile(gclip,99):5.2f} (was {np.percentile(base,99):5.2f}) "
              f"| any-layer-clipped={100*any_clip.mean():5.1f}% | mean-norm-shrink={shrink:4.1f}%")

# per-layer honest p99 (for candidate C)
print("\n=== per-layer honest grad-norm p99 (pooled over gammas 0.33 & 0.0) ===")
for L in LAYERS:
    v = np.concatenate([d[f"g{g}_{L}"].ravel() for g in (0.33,0.0)])
    vw = np.concatenate([d[f"g{g}_w{L}"] for g in (0.33,0.0)])
    print(f"  {L}: grad p99={np.percentile(v,99):6.2f}  (||W|| median={np.median(vw):6.2f})")
