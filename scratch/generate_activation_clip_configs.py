"""
Generates the 10 sweep configs for the gradient-preserving activation-clipping /
adaptive client-norm-clipping study (see plan:
now-i-want-to-peaceful-babbage.md), one JSON per model/mechanism variant, matching
the structure of configs/archive/cnn_clipped_heatmap_sweep.json but with the full
4-aggregator sweep (GM, CenteredClipping, TrMean, MultiKrum) used by the existing
cnn_mnist_clipping_1/2/4 family, so results are directly comparable.
"""
import json
import os

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(WORKSPACE_DIR, "configs", "activation_clip")

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


def make_config(model_name, results_subdir, honest_clients_extra=None):
    honest_clients = {
        "momentum": 0.9,
        "weight_decay": 0.0001,
        "batch_size": 128,
    }
    if honest_clients_extra:
        honest_clients.update(honest_clients_extra)

    return {
        "benchmark_config": dict(BASE_BENCHMARK_CONFIG),
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
        "aggregator": BASE_AGGREGATORS,
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
    # Adaptive per-coordinate quantile clip: plain (true clamp derivative)
    "cnn_mnist_clip_qcoord_plain_080": make_config(
        "cnn_mnist_clip_qcoord_plain_080", "cnn_mnist_clip_qcoord_plain_080"
    ),
    "cnn_mnist_clip_qcoord_plain_090": make_config(
        "cnn_mnist_clip_qcoord_plain_090", "cnn_mnist_clip_qcoord_plain_090"
    ),
    # Adaptive per-coordinate quantile clip: STE backward
    "cnn_mnist_clip_qcoord_ste_080": make_config(
        "cnn_mnist_clip_qcoord_ste_080", "cnn_mnist_clip_qcoord_ste_080"
    ),
    "cnn_mnist_clip_qcoord_ste_090": make_config(
        "cnn_mnist_clip_qcoord_ste_090", "cnn_mnist_clip_qcoord_ste_090"
    ),
    # Adaptive client-side gradient-norm clip (plain ReLU cnn_mnist + windowed quantile clip)
    "cnn_mnist_qclip_070": make_config(
        "cnn_mnist", "cnn_mnist_qclip_070",
        honest_clients_extra={"grad_clip_quantile": 0.70, "grad_clip_window": 100},
    ),
    "cnn_mnist_qclip_080": make_config(
        "cnn_mnist", "cnn_mnist_qclip_080",
        honest_clients_extra={"grad_clip_quantile": 0.80, "grad_clip_window": 100},
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
