import time
import os
import csv
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor

from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
from byzfl.benchmark.managers import ParamsManager, FileManager, get_snn_suffix, retry_on_error
from byzfl.benchmark.data import load_and_split_data
from byzfl.utils.gradient_geometry import compute_layer_boundaries, compute_geometry_metrics
from byzfl.utils.interarch_metrics import compute_interarch_metrics
from byzfl.utils.gradient_structure_metrics import compute_gradient_structure_metrics, pick_fixed_subset


# ===================== Sparsity Metric Functions =====================

def compute_hoyer_sparsity(x):
    """Hoyer sparsity: 0 = uniform, 1 = maximally sparse."""
    d = x.numel()
    l1 = x.abs().sum()
    l2 = x.norm()
    if l2 < 1e-12:
        return 0.0
    sqrt_d = d ** 0.5
    return ((sqrt_d - l1 / l2) / (sqrt_d - 1)).item()

def compute_gini_index(x):
    """Gini index: 0 = uniform, ~1 = sparse."""
    abs_x = x.abs()
    sorted_x = torch.sort(abs_x.flatten())[0]
    d = sorted_x.numel()
    total = sorted_x.sum()
    if total < 1e-12:
        return 0.0
    indices = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    return ((2 * (indices * sorted_x).sum()) / (d * total) - (d + 1) / d).item()

def compute_l1_l2_ratio(x):
    """Normalized L1/L2 ratio: L1/(L2*sqrt(d)). 1 = uniform, 0 = sparse."""
    d = x.numel()
    l1 = x.abs().sum()
    l2 = x.norm()
    if l2 < 1e-12:
        return 0.0
    return (l1 / (l2 * (d ** 0.5))).item()

def compute_near_zero_fraction(x, threshold):
    """Fraction of elements with absolute value below threshold."""
    return (x.abs() < threshold).float().mean().item()

def compute_kurtosis(x):
    """Excess kurtosis of the gradient distribution."""
    mean = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0
    return (((x - mean) / std) ** 4).mean().item() - 3.0

def compute_topk_concentration(x, fraction):
    """Fraction of total L1 norm concentrated in the top-k% largest elements."""
    abs_x = x.abs().flatten()
    total = abs_x.sum()
    if total < 1e-12:
        return 0.0
    k = max(1, int(abs_x.numel() * fraction))
    topk_vals = torch.topk(abs_x, k).values
    return (topk_vals.sum() / total).item()

def compute_entropy(x, num_bins=100):
    """Normalized Shannon entropy of |x| distribution (binned). 1 = uniform, 0 = concentrated."""
    if not torch.isfinite(x).all():
        return float('nan')
    abs_x = x.abs().flatten()
    if abs_x.max() < 1e-12:
        return 0.0
    # Bin the absolute values
    hist = torch.histc(abs_x, bins=num_bins, min=0, max=abs_x.max().item())
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # remove zero bins
    entropy = -(probs * probs.log()).sum().item()
    max_entropy = np.log(num_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0

def compute_all_sparsity_metrics(gradient):
    """Compute all sparsity metrics for a single gradient vector."""
    return {
        'hoyer': compute_hoyer_sparsity(gradient),
        'gini': compute_gini_index(gradient),
        'l1_l2_ratio': compute_l1_l2_ratio(gradient),
        'near_zero_1e5': compute_near_zero_fraction(gradient, 1e-5),
        'near_zero_1e3': compute_near_zero_fraction(gradient, 1e-3),
        'kurtosis': compute_kurtosis(gradient),
        'top1_concentration': compute_topk_concentration(gradient, 0.01),
        'top5_concentration': compute_topk_concentration(gradient, 0.05),
        'top10_concentration': compute_topk_concentration(gradient, 0.10),
        'entropy': compute_entropy(gradient),
    }

# =====================================================================

def start_training(params):
    params_manager = ParamsManager(params)

    # <----------------- File Manager  ----------------->
    file_manager = FileManager({
        "result_path": params_manager.get_results_directory(),
        "dataset_name": params_manager.get_dataset_name(),
        "model_name": params_manager.get_model_name(),
        "nb_workers": params_manager.get_nb_workers(),
        "nb_byz": params_manager.get_f(),
        "declared_nb_byz": params_manager.get_tolerated_f(),
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": (
            None if params_manager.get_name_data_distribution() 
            in ["iid", "extreme_niid"] 
            else params_manager.get_parameter_data_distribution()
        ),
        "aggregation_name": params_manager.get_aggregator_name(),
        "pre_aggregation_names": [
            dict['name'] 
            for dict in params_manager.get_preaggregators()
        ],
        "attack_name": params_manager.get_attack_name(),
        "learning_rate": params_manager.get_learning_rate(),
        "momentum": params_manager.get_honest_clients_momentum(),
        "weight_decay": params_manager.get_honest_clients_weight_decay(),
        "snn_suffix": get_snn_suffix(params_manager),
        "clean_directory_structure": params.get("evaluation_and_results", {}).get("clean_directory_structure", False),
        "encoding_type": params_manager.get_encoding_type(),
    })

    file_manager.save_config_dict(params_manager.get_data())

    # <----------------- Federated Framework ----------------->

    # Configurations
    nb_honest_clients = params_manager.get_nb_honest_clients()
    nb_byz_clients = params_manager.get_f()
    nb_training_steps = params_manager.get_nb_steps()
    batch_size = params_manager.get_honest_clients_batch_size()

    dd_seed = params_manager.get_data_distribution_seed()
    training_seed = params_manager.get_training_seed()
    clean = params.get("evaluation_and_results", {}).get("clean_directory_structure", False)
    if clean:
        val_acc_filename = "val_accuracy.txt"
        test_acc_filename = "test_accuracy.txt"
        train_loss_filename = "train_loss.txt"
        train_time_filename = "train_time.txt"
    else:
        val_acc_filename = f"val_accuracy_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
        test_acc_filename = f"test_accuracy_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
        train_loss_filename = f"train_loss_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
        train_time_filename = f"train_time_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt"
    set_random_seed(dd_seed)

    # Data Preparation and Splitting
    train_dataset, val_loader, test_loader = load_and_split_data(params_manager)

    print(f"\n[ByzFL-SNN] [Training Start] Model: {params_manager.get_model_name()} | Dataset: {params_manager.get_dataset_name()}")
    print(f"[ByzFL-SNN]   Aggregator: {params_manager.get_aggregator_name()} | Attack: {params_manager.get_attack_name()}")
    print(f"[ByzFL-SNN]   Distribution: {params_manager.get_name_data_distribution()} (gamma={params_manager.get_parameter_data_distribution()})")
    print(f"[ByzFL-SNN]   Clients: {nb_honest_clients} honest, {nb_byz_clients} byzantine | Algorithm: {params_manager.get_training_algorithm_name()}")
    print(f"[ByzFL-SNN]   Learning Rate: {params_manager.get_learning_rate()} | Steps: {nb_training_steps} | Device: {params_manager.get_device()}")
    print(f"[ByzFL-SNN]   Validation samples: {len(val_loader.dataset) if val_loader else 0} | Test samples: {len(test_loader.dataset) if test_loader else 0}")

    # Distribute data among clients using non-IID Dirichlet distribution
    data_distributor = DataDistributor({
        "data_distribution_name": params_manager.get_name_data_distribution(),
        "distribution_parameter": params_manager.get_parameter_data_distribution(),
        "nb_honest": nb_honest_clients,
        "data_loader": train_dataset,
        "batch_size": batch_size,
    })
    client_dataloaders = data_distributor.split_data()

    # Prepare model parameters (including time_steps for SNNs)
    model_params = params_manager.get_model_params().copy()
    if params_manager.is_snn():
        model_params["time_steps"] = params_manager.get_time_steps()

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": params_manager.get_model_name(),
            "model_params": model_params,
            "device": params_manager.get_device(),
            "optimizer_name": params_manager.get_optimizer_name(),
            "learning_rate": params_manager.get_learning_rate(),
            "loss_name": params_manager.get_loss_name(),
            "loss_params": params_manager.get_loss_params(),
            "accuracy_name": params_manager.get_accuracy_name(),
            "weight_decay": params_manager.get_honest_clients_weight_decay(),
            "milestones": params_manager.get_milestones(),
            "learning_rate_decay": params_manager.get_learning_rate_decay(),
            "LabelFlipping": "LabelFlipping" == params_manager.get_attack_name(),
            "training_dataloader": client_dataloaders[i],
            "momentum": params_manager.get_honest_clients_momentum(),
            "nb_labels": params_manager.get_nb_labels(),
            "store_per_client_metrics": params_manager.get_store_per_client_metrics(),
            "gradient_clip_val": params_manager.get_honest_clients_gradient_clip_val(),
        }) for i in range(nb_honest_clients)
    ]

    # Server Setup, Use SGD Optimizer
    server = Server({
        "model_name": params_manager.get_model_name(),
        "model_params": model_params,
        "device": params_manager.get_device(),
        "validation_loader": val_loader,
        "test_loader": test_loader,
        "accuracy_name": params_manager.get_accuracy_name(),
        "optimizer_name": params_manager.get_optimizer_name(),
        "learning_rate": params_manager.get_learning_rate(),
        "weight_decay": params_manager.get_honest_clients_weight_decay(),
        "milestones": params_manager.get_milestones(),
        "learning_rate_decay": params_manager.get_learning_rate_decay(),
        "aggregator_info": params_manager.get_aggregator_info(),
        "pre_agg_list": params_manager.get_preaggregators(),
    })

    # Byzantine Client Setup

    attack_parameters = params_manager.get_attack_parameters()
    attack_parameters["aggregator_info"] = params_manager.get_aggregator_info()
    attack_parameters["pre_agg_list"] = params_manager.get_preaggregators()
    attack_parameters["f"] = nb_byz_clients

    label_flipping_attack = False
    attack_name = params_manager.get_attack_name()

    label_flipping_attack = attack_name == "LabelFlipping"

    attack = {
        "name": attack_name,
        "f": nb_byz_clients,
        "parameters": attack_parameters,
    }
    byz_client = ByzantineClient(attack)

    byzantine_removal_step = params_manager.get_byzantine_removal_step()
    byz_removed = False

    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    store_models = params_manager.get_store_models()
    store_per_client_metrics = params_manager.get_store_per_client_metrics()
    store_client_vectors = params_manager.get_store_client_vectors()
    store_gradient_structure_metrics = params_manager.get_store_gradient_structure_metrics()

    val_accuracy_list = np.array([])
    test_accuracy_list = np.array([])
    train_loss_list = np.zeros((nb_training_steps))

    trace_var_list = np.zeros((nb_training_steps))
    norm_var_list = np.zeros((nb_training_steps))
    mean_grad_norm_list = np.zeros((nb_training_steps))
    normalized_trace_var_list = np.zeros((nb_training_steps))
    normalized_norm_var_list = np.zeros((nb_training_steps))
    
    max_deviation_list = np.zeros((nb_training_steps))
    mean_cos_sim_list = np.zeros((nb_training_steps))
    max_abs_grad_list = np.zeros((nb_training_steps))
    grad_norm_min_list = np.zeros((nb_training_steps))
    grad_norm_max_list = np.zeros((nb_training_steps))
    grad_norm_std_list = np.zeros((nb_training_steps))

    # EXP2: firing rate (honest clients' local forward pass, SNN only) and
    # effective gradient norm (server's actual post-attack, post-aggregation update).
    firing_rate_list = np.full((nb_training_steps), np.nan)
    effective_grad_norm_list = np.zeros((nb_training_steps))

    # Threshold sweep: per-layer mean firing rate (honest clients), snapshotted
    # every evaluation_delta steps, and the raw per-client vectors actually
    # sent to aggregation (post-momentum, pre-pre-aggregation), snapshotted
    # every 100 steps so sign agreement can be computed post-hoc.
    per_layer_firing_rate_rows = []
    client_vector_snapshot_steps = []
    client_vector_snapshot_dir = None

    # Gradient-geometry baseline (f=0 correlation study): online per-step
    # consensus/dispersion/sign-agreement metrics on the honest post-momentum
    # vectors, no raw vectors ever logged.
    geometry_layer_boundaries = compute_layer_boundaries(honest_clients[0].model)
    geometry_rows = []

    # Gradient-structure study (SNN-vs-CNN aggregator design, f=0 baseline):
    # online per-step PCA/active-coordinate/support-overlap metrics, same
    # no-raw-vectors-on-disk discipline as the geometry metrics above.
    gradient_structure_rows = []
    gradient_structure_subset_idx = (
        pick_fixed_subset(nb_honest_clients) if store_gradient_structure_metrics else None
    )

    # Inter-architecture metrics (N, CV_eff, cos, participation, per-layer)
    # computed on both pre-momentum (raw) and post-momentum vectors every step.
    interarch_rows = []

    # Sparsity metric arrays
    sparsity_metric_names = [
        'hoyer', 'gini', 'l1_l2_ratio', 'near_zero_1e5', 'near_zero_1e3',
        'kurtosis', 'top1_concentration', 'top5_concentration', 'top10_concentration', 'entropy'
    ]
    sparsity_mean = {name: np.zeros(nb_training_steps) for name in sparsity_metric_names}
    sparsity_std = {name: np.zeros(nb_training_steps) for name in sparsity_metric_names}

    start_time = time.time()

    # Send Initial Model to All Clients
    new_model = server.get_dict_parameters()
    for client in honest_clients:
        client.set_model_state(new_model)
    
    training_algorithm_name = params_manager.get_training_algorithm_name()

    if training_algorithm_name not in ["DSGD", "FedAvg"]:
        raise ValueError(f"Training algorithm {training_algorithm_name} not supported, supported algorithms are 'DSGD' and 'FedAvg'")
    
    if training_algorithm_name == "FedAvg" and attack_name == "LabelFlipping":
        raise ValueError("FedAvg does not support Label Flipping attack.")
    
    if training_algorithm_name == "FedAvg":

        training_algorithm_parameters = params_manager.get_training_algorithm_parameters()

        proportion_selected_clients = training_algorithm_parameters["proportion_selected_clients"]
        local_steps_per_client = training_algorithm_parameters["local_steps_per_client"]
        nb_clients_to_sample = int(nb_honest_clients * proportion_selected_clients)

    init_allocated = torch.cuda.memory_allocated(device=params_manager.get_device()) / (1024**2)
    init_reserved = torch.cuda.memory_reserved(device=params_manager.get_device()) / (1024**2)
    print(f"[ByzFL-SNN] Initial VRAM (Alloc/Reserved): {init_allocated:.1f}/{init_reserved:.1f} MiB")

    # Training Loop
    for training_step in range(nb_training_steps):

        # Evaluate Global Model Every Evaluation Delta Steps
        if training_step % evaluation_delta == 0:
            val_info = ""
            test_info = ""

            if val_loader is not None:
                val_acc = server.compute_validation_accuracy()
                val_accuracy_list = np.append(val_accuracy_list, val_acc)
                file_manager.write_array_in_file(val_accuracy_list, val_acc_filename)
                val_info = f" | Val Acc: {val_acc:.4f}"

            if evaluate_on_test:
                test_acc = server.compute_test_accuracy()
                test_accuracy_list = np.append(test_accuracy_list, test_acc)
                file_manager.write_array_in_file(test_accuracy_list, test_acc_filename)
                test_info = f" | Test Acc: {test_acc:.4f}"

            allocated = torch.cuda.memory_allocated(device=server.device) / (1024**2)
            reserved = torch.cuda.memory_reserved(device=server.device) / (1024**2)
            vram_info = f" | VRAM (Alloc/Reserved): {allocated:.1f}/{reserved:.1f} MiB"

            elapsed = time.time() - start_time
            print(f"[ByzFL-SNN]   [Step {training_step}/{nb_training_steps}] ({training_step/nb_training_steps*100:.1f}%)"
                  f"{val_info}{test_info}{vram_info} | Time: {elapsed:.1f}s")

            if store_models:
                file_manager.save_state_dict(
                    server.get_dict_parameters(),
                    training_seed,
                    dd_seed,
                    training_step
                )
        
        if training_algorithm_name == "DSGD":

            if (byzantine_removal_step is not None) and (not byz_removed) and (training_step >= byzantine_removal_step):
                byz_client.f = 0
                server.robust_aggregator.aggregator.f = 0
                for pre_agg in server.robust_aggregator.pre_agg_list:
                    pre_agg.f = 0
                byz_removed = True
                print(f"[ByzFL-SNN]   >>> Byzantine attack REMOVED at step {training_step} <<<")

            train_loss_per_client = np.zeros((nb_honest_clients))

            # Honest Clients Compute Gradients
            for i, client in enumerate(honest_clients):
                train_loss_per_client[i] = client.compute_gradients()
            
            train_loss_list[training_step] = train_loss_per_client.mean()
            
            # Capture raw gradients BEFORE momentum (for pre-momentum metrics)
            raw_gradients = [client.get_flat_gradients() for client in honest_clients]

            # Aggregate Honest Gradients (applies momentum in-place)
            honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

            # Gradient-geometry baseline: online consensus/dispersion/sign-agreement
            # metrics on the honest post-momentum vectors, every step, no vectors logged.
            geometry_row = compute_geometry_metrics(honest_gradients, geometry_layer_boundaries)
            geometry_row["step"] = training_step
            layer_firing_rates = defaultdict(list)
            for c in honest_clients:
                for layer_name, rate in c.get_last_layer_firing_rates().items():
                    layer_firing_rates[layer_name].append(rate)
            for layer_name, rates in layer_firing_rates.items():
                geometry_row[f"fr_{layer_name}"] = np.mean(rates)
            geometry_rows.append(geometry_row)

            # Inter-architecture metrics: post-momentum + pre-momentum
            firing_rates_dict = {ln: np.mean(rs) for ln, rs in layer_firing_rates.items()}
            interarch_row = compute_interarch_metrics(
                honest_gradients, geometry_layer_boundaries, firing_rates_dict, prefix=""
            )
            interarch_row.update(compute_interarch_metrics(
                raw_gradients, geometry_layer_boundaries, {}, prefix="g_"
            ))
            interarch_row["step"] = training_step
            interarch_rows.append(interarch_row)
            del raw_gradients  # free memory immediately

            # EXP2: mean firing rate across honest clients (SNN only; NaN for non-spiking models)
            client_firing_rates = [c.get_last_firing_rate() for c in honest_clients]
            client_firing_rates = [r for r in client_firing_rates if r is not None]
            if client_firing_rates:
                firing_rate_list[training_step] = np.mean(client_firing_rates)

            # Threshold sweep: per-layer mean firing rate across honest clients,
            # snapshotted every evaluation_delta steps.
            if training_step % evaluation_delta == 0:
                layer_rates = defaultdict(list)
                for c in honest_clients:
                    for layer_name, rate in c.get_last_layer_firing_rates().items():
                        layer_rates[layer_name].append(rate)
                if layer_rates:
                    row = {"step": training_step}
                    row.update({name: np.mean(vals) for name, vals in layer_rates.items()})
                    per_layer_firing_rate_rows.append(row)

            # Threshold sweep: dump the per-client vectors actually sent to
            # aggregation (post-momentum, pre-pre-aggregation), honest clients
            # only, every 100 steps. Opt-in only (store_client_vectors,
            # default False) -- these snapshots are ~nb_honest_clients x d
            # floats each (tens of MB per run for this model), so leaving
            # this on by default across a full sweep exhausts disk fast.
            if store_client_vectors and training_step % 100 == 0:
                if client_vector_snapshot_dir is None:
                    vec_subdir = (
                        f"client_vectors_tr_seed_{training_seed}_dd_seed_{dd_seed}"
                        if not clean else "client_vectors"
                    )
                    client_vector_snapshot_dir = os.path.join(file_manager.get_experiment_path(), vec_subdir)
                    os.makedirs(client_vector_snapshot_dir, exist_ok=True)
                stacked = torch.stack(honest_gradients, dim=0).detach().cpu().numpy()
                np.save(os.path.join(client_vector_snapshot_dir, f"step_{training_step}.npy"), stacked)
                client_vector_snapshot_steps.append(training_step)

            # Gradient-structure study: online PCA/active-coordinate/support-
            # overlap metrics, no vectors ever written to disk (see
            # gradient_structure_metrics.py). Sampled at evaluation_delta
            # cadence (cheap either way -- rows are tiny scalars, not the
            # raw-vector-dump's (10, d) arrays) so it lines up with the
            # existing accuracy-evaluation steps for easy overlay.
            if store_gradient_structure_metrics and training_step % evaluation_delta == 0:
                gs_row = compute_gradient_structure_metrics(honest_gradients, gradient_structure_subset_idx)
                gs_row["step"] = training_step
                gradient_structure_rows.append(gs_row)

            # Compute Variance Metrics
            if len(honest_gradients) > 0:
                stacked_grads = torch.stack(honest_gradients, dim=0) # Shape: (nb_honest_clients, d)
                mean_grad = stacked_grads.mean(dim=0)
                var_grad = stacked_grads.var(dim=0, unbiased=False)
                
                trace_var = var_grad.sum().item()
                norm_var = var_grad.norm().item()
                mean_grad_sq_norm = mean_grad.square().sum().item()
                mean_grad_norm = mean_grad_sq_norm ** 0.5
                
                trace_var_list[training_step] = trace_var
                norm_var_list[training_step] = norm_var
                mean_grad_norm_list[training_step] = mean_grad_norm
                normalized_trace_var_list[training_step] = trace_var / (mean_grad_sq_norm + 1e-9)
                normalized_norm_var_list[training_step] = norm_var / (mean_grad_sq_norm + 1e-9)

                # New robustness metrics
                norms = stacked_grads.norm(dim=1)
                grad_norm_min_list[training_step] = norms.min().item()
                grad_norm_max_list[training_step] = norms.max().item()
                grad_norm_std_list[training_step] = norms.std().item()

                deviations = (stacked_grads - mean_grad).norm(dim=1)
                max_deviation_list[training_step] = deviations.max().item()

                if mean_grad_norm > 1e-12:
                    cos_sims = (stacked_grads * mean_grad).sum(dim=1) / (norms * mean_grad_norm + 1e-12)
                    mean_cos_sim_list[training_step] = cos_sims.mean().item()
                else:
                    mean_cos_sim_list[training_step] = 0.0

                max_abs_grad_list[training_step] = stacked_grads.abs().max().item()

                # Compute sparsity metrics per client (COMMENTED OUT TO SPEED UP TRAINING)
                # all_client_metrics = [compute_all_sparsity_metrics(g) for g in honest_gradients]
                # for metric_name in sparsity_metric_names:
                #     vals = np.array([m[metric_name] for m in all_client_metrics])
                #     sparsity_mean[metric_name][training_step] = vals.mean()
                #     sparsity_std[metric_name][training_step] = vals.std()

            # Deal with Label Flipping Attack
            attack_input = (
                [client.get_flat_flipped_gradients() for client in honest_clients]
                if label_flipping_attack
                else honest_gradients
            )

            # Apply Byzantine Attack
            byz_vector = byz_client.apply_attack(attack_input)

            # Combine Honest and Byzantine Gradients
            gradients = honest_gradients + byz_vector

            # Update Global Model
            server.update_model_with_gradients(gradients)
            effective_grad_norm_list[training_step] = server.last_aggregate_grad_norm

        elif training_algorithm_name == "FedAvg":

            idx_selected_clients = np.random.choice(
                range(nb_honest_clients + nb_byz_clients), 
                size=int(nb_clients_to_sample), 
                replace=False
            )

            idx_honest_clients = idx_selected_clients[idx_selected_clients < nb_honest_clients]
            count_byz_clients = len(idx_selected_clients) - len(idx_honest_clients)
            
            train_loss_per_client = np.zeros((len(idx_honest_clients)))
            honest_weights = []

            for idx, i in enumerate(idx_honest_clients):
                train_loss_per_client[idx] = honest_clients[i].compute_model_update(local_steps_per_client)
                honest_weights.append(honest_clients[i].get_flat_parameters())
            
            train_loss_list[training_step] = train_loss_per_client.mean()

            byz_client.f = count_byz_clients
            byz_weights = byz_client.apply_attack(honest_weights)

            weights = honest_weights + byz_weights

            server.update_model_with_weights(weights)

        else:
            raise ValueError(f"Training algorithm {training_algorithm_name} not supported")
        
        # Send Updated Model to Clients
        new_model = server.get_dict_parameters()
        for client in honest_clients:
            client.set_model_state(new_model)
    
    end_time = time.time()

    file_manager.write_array_in_file(train_loss_list, train_loss_filename)

    honest_var_trace_filename = f"honest_var_trace_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_var_trace.txt"
    file_manager.write_array_in_file(trace_var_list, honest_var_trace_filename)
    
    honest_var_norm_filename = f"honest_var_norm_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_var_norm.txt"
    file_manager.write_array_in_file(norm_var_list, honest_var_norm_filename)
    
    honest_mean_grad_norm_filename = f"honest_mean_grad_norm_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_mean_grad_norm.txt"
    file_manager.write_array_in_file(mean_grad_norm_list, honest_mean_grad_norm_filename)
    
    honest_normalized_trace_var_filename = f"honest_normalized_trace_var_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_normalized_trace_var.txt"
    file_manager.write_array_in_file(normalized_trace_var_list, honest_normalized_trace_var_filename)
    
    honest_normalized_norm_var_filename = f"honest_normalized_norm_var_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_normalized_norm_var.txt"
    file_manager.write_array_in_file(normalized_norm_var_list, honest_normalized_norm_var_filename)
    
    honest_max_deviation_filename = f"honest_max_deviation_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_max_deviation.txt"
    file_manager.write_array_in_file(max_deviation_list, honest_max_deviation_filename)
    
    honest_mean_cos_sim_filename = f"honest_mean_cos_sim_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_mean_cos_sim.txt"
    file_manager.write_array_in_file(mean_cos_sim_list, honest_mean_cos_sim_filename)
    
    honest_max_abs_grad_filename = f"honest_max_abs_grad_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_max_abs_grad.txt"
    file_manager.write_array_in_file(max_abs_grad_list, honest_max_abs_grad_filename)
    
    honest_grad_norm_min_filename = f"honest_grad_norm_min_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_grad_norm_min.txt"
    file_manager.write_array_in_file(grad_norm_min_list, honest_grad_norm_min_filename)
    
    honest_grad_norm_max_filename = f"honest_grad_norm_max_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_grad_norm_max.txt"
    file_manager.write_array_in_file(grad_norm_max_list, honest_grad_norm_max_filename)
    
    honest_grad_norm_std_filename = f"honest_grad_norm_std_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_grad_norm_std.txt"
    file_manager.write_array_in_file(grad_norm_std_list, honest_grad_norm_std_filename)

    honest_firing_rate_filename = f"honest_firing_rate_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "honest_firing_rate.txt"
    file_manager.write_array_in_file(firing_rate_list, honest_firing_rate_filename)

    effective_grad_norm_filename = f"effective_grad_norm_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else "effective_grad_norm.txt"
    file_manager.write_array_in_file(effective_grad_norm_list, effective_grad_norm_filename)

    if per_layer_firing_rate_rows:
        layer_names_sorted = sorted({
            key for row in per_layer_firing_rate_rows for key in row.keys() if key != "step"
        })
        layer_firing_rate_filename = (
            f"layer_firing_rate_tr_seed_{training_seed}_dd_seed_{dd_seed}.csv" if not clean else "layer_firing_rate.csv"
        )
        layer_firing_rate_path = os.path.join(file_manager.get_experiment_path(), layer_firing_rate_filename)
        def do_write_layer_firing_rate():
            with open(layer_firing_rate_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["step"] + layer_names_sorted)
                writer.writeheader()
                for row in per_layer_firing_rate_rows:
                    writer.writerow(row)
        retry_on_error(do_write_layer_firing_rate)

    if geometry_rows:
        geometry_fieldnames = sorted({key for row in geometry_rows for key in row.keys() if key != "step"})
        geometry_filename = (
            f"metrics_geometry_tr_seed_{training_seed}_dd_seed_{dd_seed}.csv" if not clean else "metrics_geometry.csv"
        )
        geometry_path = os.path.join(file_manager.get_experiment_path(), geometry_filename)
        def do_write_geometry():
            with open(geometry_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["step"] + geometry_fieldnames)
                writer.writeheader()
                for row in geometry_rows:
                    writer.writerow(row)
        retry_on_error(do_write_geometry)

    if interarch_rows:
        interarch_fieldnames = sorted({key for row in interarch_rows for key in row.keys() if key != "step"})
        interarch_filename = (
            f"metrics_interarch_tr_seed_{training_seed}_dd_seed_{dd_seed}.csv" if not clean else "metrics_interarch.csv"
        )
        interarch_path = os.path.join(file_manager.get_experiment_path(), interarch_filename)
        def do_write_interarch():
            with open(interarch_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["step"] + interarch_fieldnames)
                writer.writeheader()
                for row in interarch_rows:
                    writer.writerow(row)
        retry_on_error(do_write_interarch)

    if gradient_structure_rows:
        gs_fieldnames = sorted({key for row in gradient_structure_rows for key in row.keys() if key != "step"})
        gs_filename = (
            f"metrics_gradient_structure_tr_seed_{training_seed}_dd_seed_{dd_seed}.csv"
            if not clean else "metrics_gradient_structure.csv"
        )
        gs_path = os.path.join(file_manager.get_experiment_path(), gs_filename)
        def do_write_gradient_structure():
            with open(gs_path, "w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=["step"] + gs_fieldnames)
                writer.writeheader()
                for row in gradient_structure_rows:
                    writer.writerow(row)
        retry_on_error(do_write_gradient_structure)

    # Save sparsity metrics
    for metric_name in sparsity_metric_names:
        mean_fn = f"sparsity_{metric_name}_mean_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else f"sparsity_{metric_name}_mean.txt"
        std_fn = f"sparsity_{metric_name}_std_tr_seed_{training_seed}_dd_seed_{dd_seed}.txt" if not clean else f"sparsity_{metric_name}_std.txt"
        file_manager.write_array_in_file(sparsity_mean[metric_name], mean_fn)
        file_manager.write_array_in_file(sparsity_std[metric_name], std_fn)

    if val_loader is not None:
    
        val_acc = server.compute_validation_accuracy()

        val_accuracy_list = np.append(val_accuracy_list, val_acc)

        file_manager.write_array_in_file(val_accuracy_list, val_acc_filename)

    if evaluate_on_test:
        test_acc = server.compute_test_accuracy()
        test_accuracy_list = np.append(test_accuracy_list, test_acc)

        file_manager.write_array_in_file(test_accuracy_list, test_acc_filename)

    if store_per_client_metrics:

        for client_id, client in enumerate(honest_clients):
            loss = client.get_loss_list()
            acc = client.get_train_accuracy()
            
            file_manager.save_loss(
                loss,
                training_seed,
                dd_seed,
                client_id
            )
            
            file_manager.save_accuracy(
                acc,
                training_seed,
                dd_seed,
                client_id
            )
    
    if store_models:
        file_manager.save_state_dict(
            server.get_dict_parameters(),
            training_seed,
            dd_seed,
            training_step
        )
    
    execution_time = end_time - start_time

    file_manager.write_array_in_file(np.array(execution_time), train_time_filename)
    
    peak_allocated = torch.cuda.max_memory_allocated(device=server.device) / (1024**2)
    peak_reserved = torch.cuda.max_memory_reserved(device=server.device) / (1024**2)
    print(f"[ByzFL-SNN] [Training Complete] Peak VRAM (Alloc/Reserved): {peak_allocated:.1f}/{peak_reserved:.1f} MiB | Total time: {execution_time:.1f}s\n")

    # Clean up variables to free memory
    import gc
    del server
    del honest_clients
    del byz_client
    del train_dataset
    del val_loader
    del test_loader
    if 'client_dataloaders' in locals():
        del client_dataloaders
    gc.collect()
    torch.cuda.empty_cache()