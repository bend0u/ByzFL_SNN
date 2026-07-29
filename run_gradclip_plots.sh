#!/usr/bin/env bash
#
# Regenerate the grad-clip heatmaps ONCE, after BOTH pods (A and B) have finished
# and their gamma-halves have merged into each config's results_directory. The
# plotter is read-only (reads raw per-setting result files), so this is safe.
#
# IMPORTANT: POD A and POD B each ran with a gamma-SUBSET config (--gammas 1.0 0.66
# / --gammas 0.33 0.0), and run_benchmark() overwrites results_directory/config.json
# with whatever (normalized) config it was given -- so after both pods finish,
# config.json holds only the LAST pod's 2-gamma subset, not the full 4. The heatmap
# code reads its gamma list FROM that file, so without fixing it first, heatmaps
# would silently show only 2 of 4 rows (no error). The per-setting result .txt files
# themselves are fine (both halves are correctly on disk) -- only this metadata
# needs restoring, and it must be a FIELD PATCH (not "cp the source config back"):
# run_benchmark() fills in defaults (e.g. set_honest_clients_as_clients) via
# ensure_optional_config_parameters() before writing config.json, and the raw
# source config in configs/activation_clip/ does NOT have those defaults -- copying
# it wholesale strips them and breaks the heatmap reader (KeyError). So we only
# overwrite the gamma list field, in place, preserving everything else already
# normalized in the on-disk config.json.
#
# Run inside the RCP container from the repo root, or on the SSH box (add
# PYTHONPATH=. there). Cheap (no training) -- can run as a tiny 1-GPU/CPU job.
# Usage:  bash run_gradclip_plots.sh

set -e

CONFIGS=(
  cnn_mnist_gradclip21
  cnn_mnist_gradclip_calib
  cnn_mnist_gradclip_qlow
  cnn_mnist_gradclip_qhigh
  cnn_mnist_layerclip
)

echo "Restoring full 4-gamma list in each results_directory/config.json (undoing"
echo "the per-pod gamma-subset overwrite; all other fields untouched)..."
python3 - "${CONFIGS[@]}" <<'PYEOF'
import json, os, sys

FULL_GAMMAS = [1.0, 0.66, 0.33, 0.0]

for name in sys.argv[1:]:
    src = f"configs/activation_clip/{name}.json"
    dst = f"results/activation_clip/{name}/config.json"
    if not os.path.isfile(dst):
        print(f"  skip {name} (no results yet)")
        continue
    with open(dst) as f:
        cfg = json.load(f)
    with open(src) as f:
        src_cfg = json.load(f)
    expected = src_cfg["benchmark_config"]["data_distribution"][0]["distribution_parameter"]
    cfg["benchmark_config"]["data_distribution"][0]["distribution_parameter"] = expected
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=4, separators=(",", ": "))
    print(f"  patched {dst} -> gammas {expected}")
PYEOF

python plot_activation_clip_results.py --workers 5 --configs "${CONFIGS[@]}"
echo "Heatmaps regenerated under results/activation_clip_plots/<config>/."
