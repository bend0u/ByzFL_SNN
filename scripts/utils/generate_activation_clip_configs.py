"""
Generates the 9 sweep configs for the gradient-preserving activation-clipping /
adaptive client-norm-clipping study, one JSON per model/mechanism variant, matching
the structure of configs/archive/cnn_clipped_heatmap_sweep.json but with the full
4-aggregator sweep (GM, CenteredClipping, TrMean, MultiKrum) used by the existing
cnn_mnist_clipping_1/2/4 family, so results are directly comparable.
"""
import json
import os

# This file lives at <repo>/scripts/utils/, so the repo root is three levels up.
WORKSPACE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
OUT_DIR = os.path.join(WORKSPACE_DIR, "configs", "activation_clip")

# Calibrated clip thresholds emitted by scripts/experiments/calibrate_clip.py
# (the "smallest absolute cap leaving >=q of honest updates untouched at every
# gamma" rule). If absent, fall back to hand values documented from the probe.
_CALIB_PATH = os.path.join(WORKSPACE_DIR, "results", "activation_clip",
                           "gradnorm_probe", "clip_calibration.json")
if os.path.exists(_CALIB_PATH):
    with open(_CALIB_PATH) as _f:
        _CALIB = json.load(_f)
    CALIB_GLOBAL = _CALIB["global_clip"]              # headline q (0.99)
    CALIB_LAYERS = _CALIB["layer_clip"]               # per-layer q0.99 vector
    CALIB_BY_Q = _CALIB["global_clip_by_q"]           # {"0.9":.., "0.99":.., "1":..}
else:
    # Fallbacks (used only if calibrate_clip.py has not been run yet); the real
    # numbers all come from clip_calibration.json.
    CALIB_GLOBAL = 17.6
    CALIB_LAYERS = {"_c1": 1.3, "_c2": 4.4, "_f1": 12.3, "_f2": 12.1}
    CALIB_BY_Q = {"0.9": 9.0, "0.99": 17.6, "1": 27.0}

# The two value-sweep brackets = the tightest and loosest algorithm quantiles.
_qs = sorted(CALIB_BY_Q, key=float)
CALIB_BRACKET_LOW = CALIB_BY_Q[_qs[0]]                 # e.g. q0.90
CALIB_BRACKET_HIGH = CALIB_BY_Q[_qs[-1]]              # e.g. q1.0 (max)

BASE_BENCHMARK_CONFIG = {
    "device": "cuda",
    "training_seed": 42,
    "nb_training_seeds": 5,
    "nb_honest_clients": 10,
    "f": [0, 1, 2, 3, 4, 5],
    "size_train_set": 0.8,
    "data_distribution_seed": 42,
    "nb_data_distribution_seeds": 1,
    "data_distribution": [
        {
            "name": "gamma_similarity_niid",
            "distribution_parameter": [1.0, 0.66, 0.33, 0.0],
        }
    ],
    "training_algorithm": {"name": "DSGD", "parameters": {}},
    "nb_steps": 500,
}

BASE_AGGREGATORS = [
    {"name": "GeometricMedian", "parameters": {"nu": 0.1, "T": 3}},
    {"name": "CenteredClipping", "parameters": {}},
    {"name": "TrMean", "parameters": {}},
    {"name": "MultiKrum", "parameters": {}},
]

BASE_PRE_AGGREGATORS = [
    {"name": "NNM", "parameters": {}},
    {"name": "ARC", "parameters": {}},
]

BASE_ATTACKS = [
    {"name": "Optimal_ALittleIsEnough_neg1", "parameters": {}},
    {"name": "SignFlipping", "parameters": {}},
    {"name": "Optimal_InnerProductManipulation", "parameters": {}},
]


def make_config(model_name, results_subdir, honest_clients_extra=None,
                benchmark_extra=None, aggregators=None):
    honest_clients = {
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "batch_size": 128,
    }
    if honest_clients_extra:
        honest_clients.update(honest_clients_extra)

    benchmark_config = dict(BASE_BENCHMARK_CONFIG)
    if benchmark_extra:
        benchmark_config.update(benchmark_extra)

    return {
        "benchmark_config": benchmark_config,
        "model": {
            "name": model_name,
            "is_snn": False,
            "dataset_name": "mnist",
            "nb_labels": 10,
            "loss": "NLLLoss",
            "accuracy_name": None,
            "optimizer_name": "SGD",
            "learning_rate": 0.15,
            "learning_rate_decay": 1.0,
            "milestones": [],
        },
        "aggregator": aggregators if aggregators is not None else BASE_AGGREGATORS,
        "pre_aggregators": BASE_PRE_AGGREGATORS,
        "honest_clients": honest_clients,
        "attack": BASE_ATTACKS,
        "evaluation_and_results": {
            "evaluation_delta": 50,
            "batch_size_evaluation": 128,
            "evaluate_on_test": True,
            "clean_directory_structure": False,
            "store_models": False,
            "store_per_client_metrics": False,
            "data_folder": "./data",
            "results_directory": f"results/activation_clip/{results_subdir}",
        },
    }


CONFIGS = {
    # Fixed-clip STE variants
    "cnn_mnist_clip_ste_1": make_config("cnn_mnist_clip_ste_1", "cnn_mnist_clip_ste_1"),
    "cnn_mnist_clip_ste_2": make_config("cnn_mnist_clip_ste_2", "cnn_mnist_clip_ste_2"),
    # Fixed-clip linear-ramp variants
    "cnn_mnist_clip_ramp_1": make_config("cnn_mnist_clip_ramp_1", "cnn_mnist_clip_ramp_1"),
    "cnn_mnist_clip_ramp_2": make_config("cnn_mnist_clip_ramp_2", "cnn_mnist_clip_ramp_2"),
    # Adaptive per-coordinate quantile clip: plain (true clamp derivative).
    # The STE-backward variants were dropped: STE consistently underperformed on
    # MNIST (it hurts even clean f=0 accuracy), so it is not worth sweeping.
    "cnn_mnist_clip_qcoord_plain_080": make_config(
        "cnn_mnist_clip_qcoord_plain_080", "cnn_mnist_clip_qcoord_plain_080"
    ),
    "cnn_mnist_clip_qcoord_plain_090": make_config(
        "cnn_mnist_clip_qcoord_plain_090", "cnn_mnist_clip_qcoord_plain_090"
    ),
    # Adaptive client-side gradient-norm clip, applied to the POST-momentum vector
    # sent to the server (plain ReLU cnn_mnist + windowed quantile clip).
    "cnn_mnist_qclip_070": make_config(
        "cnn_mnist", "cnn_mnist_qclip_070",
        honest_clients_extra={"grad_clip_quantile": 0.70, "grad_clip_window": 100},
    ),
    "cnn_mnist_qclip_080": make_config(
        "cnn_mnist", "cnn_mnist_qclip_080",
        honest_clients_extra={"grad_clip_quantile": 0.80, "grad_clip_window": 100},
    ),
    # Same adaptive windowed-quantile mechanism, but applied to the RAW gradient
    # BEFORE the momentum accumulator -- the adaptive counterpart of the fixed
    # gradient_clip_val (which also clips the raw gradient), and the direct
    # comparison point against the post-momentum qclip variants above.
    # Pilot run at 2 seeds instead of 5, to get an early read on raw-gradient vs
    # momentum clipping. This is FREE relative to the full run: seeds are
    # generated as training_seed + i (42, 43, ...), so a 2-seed run is a strict
    # prefix of the 5-seed one. Bumping nb_training_seeds back to 5 later reuses
    # the cached seeds 42/43 (same results_directory) and only trains 44/45/46.
    # NOTE: 2 seeds is coarse in the collapse regions, where outcomes are close to
    # bimodal (learns vs. floors at ~0.16) -- fine for spotting a gross difference,
    # not for final numbers.
    "cnn_mnist_rawqclip_080": make_config(
        "cnn_mnist", "cnn_mnist_rawqclip_080",
        honest_clients_extra={"raw_grad_clip_quantile": 0.80, "raw_grad_clip_window": 100},
        benchmark_extra={"nb_training_seeds": 2},
    ),
    # ------------------------------------------------------------------
    # FIXED absolute raw-gradient-norm caps (no SNN, no online recalibration).
    # The offline probe (scripts/experiments/gradnorm_probe.py) shows the honest
    # cnn_mnist raw grad-norm ceiling is ~21 at gamma=0.33 (max) / ~p99 at
    # gamma=0.0, i.e. the SNN-derived 21 IS the CNN's own honest ceiling. These
    # sweeps (a) REPRODUCE fixed-21 across ALL 4 aggregators/5 seeds (the previous
    # fixed-21 run had only GeometricMedian), and (b) validate the "cap = honest
    # ceiling" rule by bracketing it tighter (10) and looser (35).
    # Reproduce the SNN-derived value 21 (kept hardcoded as the known-good
    # baseline) across all 4 aggregators / 5 seeds.
    "cnn_mnist_gradclip21": make_config(
        "cnn_mnist", "cnn_mnist_gradclip21",
        honest_clients_extra={"gradient_clip_val": 21},
    ),
    # Same, but the cap comes from the calibration ALGORITHM (calibrate_clip.py),
    # not from the SNN -- the point is that the algorithm's number ~= 21.
    "cnn_mnist_gradclip_calib": make_config(
        "cnn_mnist", "cnn_mnist_gradclip_calib",
        honest_clients_extra={"gradient_clip_val": CALIB_GLOBAL},
    ),
    # Value-sweep brackets = the tightest/loosest algorithm quantiles (q0.90 and
    # q1.0/max, worst-case over gamma), read from clip_calibration.json -- NOT
    # hand-typed. With gradclip_calib (q0.99) and the SNN's 21, this traces the
    # accuracy-vs-cap curve using only algorithm-derived points. Full 4-aggregator
    # sweep like the rest of the family so best_test is comparable.
    "cnn_mnist_gradclip_qlow": make_config(
        "cnn_mnist", "cnn_mnist_gradclip_qlow",
        honest_clients_extra={"gradient_clip_val": CALIB_BRACKET_LOW},
    ),
    "cnn_mnist_gradclip_qhigh": make_config(
        "cnn_mnist", "cnn_mnist_gradclip_qhigh",
        honest_clients_extra={"gradient_clip_val": CALIB_BRACKET_HIGH},
    ),
    # Per-layer fixed absolute caps at each layer's own offline honest ceiling
    # (~1.5x per-layer p99 from gradnorm_probe): quadrature ~= 26, comparable to
    # the global-21 anchor but distributed so no single layer can consume the
    # whole budget. Same "neuron/layer-level" idea, calibrated once offline.
    "cnn_mnist_layerclip": make_config(
        "cnn_mnist", "cnn_mnist_layerclip",
        honest_clients_extra={"layer_grad_clip_val": CALIB_LAYERS},
    ),
    # Self-calibrated warmup clip: each client freezes its OWN absolute cap from
    # its own first self_grad_clip_warmup raw grad-norms (max seen * margin), no
    # offline probe/SNN needed. w75 is the data-justified default (see gradnorm_probe
    # analysis: honest raw grad-norm peaks ~step 25-60 during the heterogeneity
    # transient, and a 75-step warmup converges to ~0% honest clipping afterward,
    # matching the offline-calibrated ceiling). w1 (literally "clip to the first
    # gradient") is a deliberate ablation: a single-sample cap sits 4.5x-8x below the
    # true honest peak and is expected to clip 34-95% of honest steps -- the same
    # over-clipping failure mode as the adaptive quantile/STE, kept to empirically
    # confirm the warmup is necessary.
    "cnn_mnist_selfclip_w75": make_config(
        "cnn_mnist", "cnn_mnist_selfclip_w75",
        honest_clients_extra={"self_grad_clip_warmup": 75, "self_grad_clip_margin": 1.1},
    ),
    "cnn_mnist_selfclip_w1": make_config(
        "cnn_mnist", "cnn_mnist_selfclip_w1",
        honest_clients_extra={"self_grad_clip_warmup": 1, "self_grad_clip_margin": 1.0},
    ),
}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for file_stem, config in CONFIGS.items():
        path = os.path.join(OUT_DIR, f"{file_stem}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=4)
            f.write("\n")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
