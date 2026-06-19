import time

import numpy as np
import torch
from torch import Tensor

from byzfl import Client, Server, ByzantineClient, DataDistributor
from byzfl.utils.misc import set_random_seed
from byzfl.benchmark.managers import ParamsManager, FileManager, get_snn_suffix
from byzfl.benchmark.data import load_and_split_data

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

    # Initialize Honest Clients
    honest_clients = [
        Client({
            "model_name": params_manager.get_model_name(),
            "model_params": params_manager.get_model_params(),
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
        }) for i in range(nb_honest_clients)
    ]

    # Server Setup, Use SGD Optimizer
    server = Server({
        "model_name": params_manager.get_model_name(),
        "model_params": params_manager.get_model_params(),
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

    set_random_seed(training_seed)

    evaluation_delta = params_manager.get_evaluation_delta()
    evaluate_on_test = params_manager.get_evaluate_on_test()

    store_models = params_manager.get_store_models()
    store_per_client_metrics = params_manager.get_store_per_client_metrics()

    val_accuracy_list = np.array([])
    test_accuracy_list = np.array([])
    train_loss_list = np.zeros((nb_training_steps))

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

            train_loss_per_client = np.zeros((nb_honest_clients))

            # Honest Clients Compute Gradients
            for i, client in enumerate(honest_clients):
                train_loss_per_client[i] = client.compute_gradients()
            
            train_loss_list[training_step] = train_loss_per_client.mean()
            
            # Aggregate Honest Gradients
            honest_gradients = [client.get_flat_gradients_with_momentum() for client in honest_clients]

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