"""
Clip-threshold CALIBRATION algorithm (no SNN, no adversary, no online oracle).

Definition
----------
The clip is the smallest ABSOLUTE cap that leaves >= q of honest updates untouched
at every heterogeneity level in the grid:

    for gamma in gammas:
        run the honest DSGD loop (f=0, all-honest averaging, real gamma-partition)
        record every raw per-client gradient norm ||g||  (BEFORE momentum)
        P[gamma] = q-quantile of the pooled ||g||
    clip = max over gamma of P[gamma]

Same rule applied per layer gives the per-layer cap vector c_L. Deterministic given
(model, data, q, steps, seeds); reproducible; uses the CNN itself.

Outputs results/activation_clip/gradnorm_probe/clip_calibration.json (consumed by
generate_activation_clip_configs.py) and prints the derivation table.
"""
import os, sys, json, argparse
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from scripts.experiments.gradnorm_probe import run_gamma, LAYERS

OUT = os.path.join(REPO, "results", "activation_clip", "gradnorm_probe")
os.makedirs(OUT, exist_ok=True)


def calibrate(gammas, q, steps, seeds, clients, lr, mom, wd, bs, device):
    # pooled[gamma] -> {"global": array, "_c1": array, ...} over steps x clients x seeds
    per_gamma = {}
    for g in gammas:
        glob, layers = [], {L: [] for L in LAYERS}
        for s in seeds:
            r = run_gamma(g, clients, steps, lr, mom, wd, bs, device, s)
            glob.append(r["global_norms"].ravel())
            for L in LAYERS:
                layers[L].append(r["layer_norms"][L].ravel())
        per_gamma[g] = {"global": np.concatenate(glob),
                        **{L: np.concatenate(layers[L]) for L in LAYERS}}
    return per_gamma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 0.33, 0.0])
    ap.add_argument("--q", type=float, default=0.99,
                    help="fraction of honest updates the cap must NOT touch")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bracket_quantiles", type=float, nargs="+", default=[0.90, 1.0],
                    help="extra global-cap quantiles emitted for the value-sweep "
                         "brackets (same worst-case-over-gamma rule as the main q)")
    args = ap.parse_args()

    pg = calibrate(args.gammas, args.q, args.steps, args.seeds, args.clients,
                   args.lr, args.mom, args.wd, args.bs, args.device)

    keys = ["global"] + LAYERS

    def worst_case(key, q):
        # cap = max over gamma of the q-quantile of the pooled honest norms
        return max(float(np.quantile(pg[g][key], q)) for g in args.gammas)

    # per-(gamma,key) q-quantile at the MAIN q (for the printed table + per_gamma dump)
    tbl = {k: {g: float(np.quantile(pg[g][k], args.q)) for g in args.gammas} for k in keys}
    cap = {k: worst_case(k, args.q) for k in keys}

    print(f"\n=== clip calibration (q={args.q}, seeds={args.seeds}, steps={args.steps}) ===")
    header = f"{'key':>7} | " + " ".join(f"g={g:>4}" for g in args.gammas) + " | worst-case cap"
    print(header)
    for k in keys:
        row = " ".join(f"{tbl[k][g]:6.2f}" for g in args.gammas)
        print(f"{k:>7} | {row} | {cap[k]:6.2f}")
    layer_caps = {L: round(cap[L], 1) for L in LAYERS}
    print(f"\nGLOBAL calibrated clip (q={args.q}) = {cap['global']:.2f}  (SNN value was 21)")
    print("PER-LAYER calibrated caps =", layer_caps,
          f" (quadrature = {np.sqrt(sum(v**2 for v in layer_caps.values())):.2f})")

    # All algorithm-derived global caps the config generator consumes (headline q
    # plus the bracket quantiles), keyed by quantile. Nothing is hand-typed.
    all_qs = sorted(set([args.q] + args.bracket_quantiles))
    global_clip_by_q = {f"{q:g}": round(worst_case("global", q), 1) for q in all_qs}
    print("GLOBAL caps by quantile (bracket set) =", global_clip_by_q)

    out = {
        "q": args.q, "seeds": args.seeds, "steps": args.steps,
        "gammas": args.gammas,
        "global_clip": round(cap["global"], 1),
        "global_clip_by_q": global_clip_by_q,
        "layer_clip": layer_caps,
        "per_gamma_quantile": {k: {str(g): round(tbl[k][g], 2) for g in args.gammas} for k in keys},
    }
    path = os.path.join(OUT, "clip_calibration.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
