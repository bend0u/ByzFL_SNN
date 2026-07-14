import json
import os

config = {
    "training_hyperparameters": {
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "milestones": [300],
        "learning_rate_decay": 0.1,
        "training_steps": 500
    },
    "evaluation_and_results": {
        "evaluation_delta": 50,
        "batch_size_evaluation": 128,
        "evaluate_on_test": True,
        "results_dir": "results/dropout_calibration"
    },
    "sweep": {
        "models": [
            "cnn_mnist_tanh_dropout_0",
            "cnn_mnist_tanh_dropout_20",
            "cnn_mnist_tanh_dropout_40",
            "cnn_mnist_tanh_dropout_60",
            "cnn_mnist_tanh_dropout_75",
            "cnn_mnist_tanh_dropout_80",
            "cnn_mnist_tanh_dropout_85",
            "cnn_mnist_tanh_dropout_90",
            "cnn_mnist_tanh_dropout_93"
        ],
        "datasets": ["mnist"],
        "nodes": [15],
        "faulty_nodes": [0],
        "data_distributions": [{"name": "gamma_similarity_niid", "gamma": 1.0}],
        "aggregators": ["Average"],
        "attacks": [{"name": "SignFlipping", "f": 0}],
        "training_seeds": [42, 43, 44, 45, 46],
        "data_distribution_seeds": [42]
    }
}

os.makedirs('configs', exist_ok=True)
with open('configs/dropout_calibration.json', 'w') as f:
    json.dump(config, f, indent=4)
