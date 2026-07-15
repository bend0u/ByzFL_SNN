import os
import datetime
import json

import numpy as np
import torch


def get_snn_suffix(params_manager):
    """
    Builds a unique directory suffix based on SNN configurations.
    """
    if not params_manager.is_snn():
        return ""
    
    parts = []
    time_steps = params_manager.get_time_steps()
    if time_steps is not None:
        parts.append(f"ts_{time_steps}")
        
    encoding = params_manager.get_encoding_type()
    if encoding is not None:
        parts.append(f"enc_{encoding}")
        
    model_params = params_manager.get_model_params()
    for k, v in sorted(model_params.items()):
        if isinstance(v, dict):
            for sub_k, sub_v in sorted(v.items()):
                if not isinstance(sub_v, (dict, list)):
                    parts.append(f"{sub_k}_{sub_v}")
        elif not isinstance(v, list):
            parts.append(f"{k}_{v}")
            
    return f"_{'_'.join(parts)}" if parts else ""



def retry_on_error(func, *args, max_retries=12, delay=5, **kwargs):
    import time
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[FileManager Retry] Failed to execute file operation on attempt {attempt + 1}/{max_retries}. Error: {e}. Retrying in {delay}s...")
            time.sleep(delay)
    return func(*args, **kwargs)


class FileManager:
    """
    Description
    -----------
    Manages the creation of directories and files to store results.
    """

    def __init__(self, params=None):
        clean = params.get("clean_directory_structure", False)
        self.clean = clean
        if clean:
            encoding = params.get("encoding_type", "constant")
            enc_name = "direct" if encoding == "constant" else encoding
            parent_dir = f"{params['dataset_name']}_{enc_name}"
            
            pre_aggs = params.get("pre_aggregation_names", [])
            agg = params.get("aggregation_name", "Average")
            preaggs_aggregator = '_'.join(pre_aggs + [agg])
            
            dist_param = params.get("distribution_parameter")
            dist_name = params.get("data_distribution_name", "iid")
            if dist_param is not None:
                dist_part = f"{dist_name}_{dist_param}"
            else:
                dist_part = dist_name

            folder_name = (
                f"{params['attack_name']}_"
                f"{preaggs_aggregator}_"
                f"f_{params['nb_byz']}_"
                f"{dist_part}"
            )
            self.files_path = f"{params['result_path']}/{parent_dir}/{folder_name}/"
        else:
            snn_suffix = params.get("snn_suffix", "")
            self.files_path = (
                f"{params['result_path']}/"
                f"{params['dataset_name']}_{params['model_name']}_"
                f"n_{params['nb_workers']}_"
                f"f_{params['nb_byz']}_"
                f"d_{params['declared_nb_byz']}_"
                f"{params['data_distribution_name']}_"
                f"{params['distribution_parameter']}_"
                f"{params['aggregation_name']}_"
                f"{'_'.join(params['pre_aggregation_names'])}_"
                f"{params['attack_name']}_"
                f"lr_{params['learning_rate']}_"
                f"mom_{params['momentum']}_"
                f"wd_{params['weight_decay']}"
                f"{snn_suffix}/"
            )
        
        def init_dirs():
            os.makedirs(self.files_path, exist_ok=True)
            with open(os.path.join(self.files_path, "day.txt"), "w") as file:
                file.write(datetime.date.today().strftime("%d_%m_%y"))

        retry_on_error(init_dirs)

    def set_experiment_path(self, path):
        """
        Set the base path for the experiment files.
        """
        self.files_path = path

    def get_experiment_path(self):
        """
        Get the current experiment path.
        """
        return self.files_path

    def save_config_dict(self, dict_to_save):
        """
        Save a configuration dictionary as a JSON file.
        """
        config_path = os.path.join(self.files_path, "config.json")
        def do_write():
            with open(config_path, "w") as json_file:
                json.dump(dict_to_save, json_file, indent=4, separators=(",", ": "))
        retry_on_error(do_write)

    def write_array_in_file(self, array, file_name):
        """
        Write a single array to a file.
        """
        file_path = os.path.join(self.files_path, file_name)
        def do_write():
            np.savetxt(file_path, [array], fmt="%.4f", delimiter=",")
        retry_on_error(do_write)

    def save_state_dict(self, state_dict, training_seed, data_dist_seed, step):
        """
        Save a model's state dictionary under a directory structured by seed values.
        """
        if self.clean:
            model_dir = os.path.join(self.files_path, "models")
        else:
            model_dir = os.path.join(
                self.files_path, f"models_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
            )
        def do_write():
            os.makedirs(model_dir, exist_ok=True)
            file_path = os.path.join(model_dir, f"model_step_{step}.pth")
            torch.save(state_dict, file_path)
        retry_on_error(do_write)

    def save_loss(self, loss_array, training_seed, data_dist_seed, client_id):
        """
        Save a loss array for a specific client and seed values.
        """
        if self.clean:
            loss_dir = os.path.join(self.files_path, "train_loss_per_client")
        else:
            loss_dir = os.path.join(
                self.files_path, f"train_loss_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
            )
        def do_write():
            os.makedirs(loss_dir, exist_ok=True)
            file_path = os.path.join(loss_dir, f"loss_client_{client_id}.txt")
            np.savetxt(file_path, loss_array, fmt="%.6f", delimiter=",")
        retry_on_error(do_write)

    def save_accuracy(self, acc_array, training_seed, data_dist_seed, client_id):
        """
        Save an accuracy array for a specific client and seed values.
        """
        if self.clean:
            acc_dir = os.path.join(self.files_path, "train_accuracy_per_client")
        else:
            acc_dir = os.path.join(
                self.files_path,
                f"train_accuracy_tr_seed_{training_seed}_dd_seed_{data_dist_seed}"
            )
        def do_write():
            os.makedirs(acc_dir, exist_ok=True)
            file_path = os.path.join(acc_dir, f"accuracy_client_{client_id}.txt")
            np.savetxt(file_path, acc_array, fmt="%.4f", delimiter=",")
        retry_on_error(do_write)




class ParamsManager(object):
    """
    Description
    -----------
    Object whose responsibility is to manage and store all the parameters
    from the JSON structure.
    """

    def __init__(self, params):
        self.data = params
        self._validate_snn_params()

    def _parameter_to_use(self, default, read):
        if read is None:
            return default
        else:
            return read

    def _read_object(self, path):
        """
        Safely traverse the nested dictionary `self.data` using the list of keys in `path`.
        Returns None if a key doesn't exist.
        """
        obj = self.data
        for p in path:
            if isinstance(obj, dict) and p in obj.keys():
                obj = obj[p]
            else:
                return None
        return obj

    def get_data(self):
        return {
            "benchmark_config": {
                "device": self.get_device(),
                "training_seed": self.get_training_seed(),
                "nb_training_seeds": self.get_nb_training_seeds(),
                "nb_workers": self.get_nb_workers(),
                "nb_honest_clients": self.get_nb_honest_clients(),
                "f": self.get_f(),
                "tolerated_f": self.get_tolerated_f(),
                "set_honest_clients_as_clients": self.get_set_honest_clients_as_clients(),
                "size_train_set": self.get_size_train_set(),
                "data_distribution_seed": self.get_data_distribution_seed(),
                "nb_data_distribution_seeds": self.get_nb_data_distribution_seeds(),
                "data_distribution": self.get_data_distribution(),
                "training_algorithm": self.get_training_algorithm(),
                "nb_steps": self.get_nb_steps()
            },
            "model": {
                "name": self.get_model_name(),
                "is_snn": self.is_snn(),
                "dataset_name": self.get_dataset_name(),
                "nb_labels": self.get_nb_labels(),
                "loss": self.get_loss_name(),
                "accuracy_name": self.get_accuracy_name(),
                "loss_params": self.get_loss_params(),
                "model_params": self.get_model_params(),
                "encoding": {
                    "type": self.get_encoding_type(),
                    "time_steps": self.get_time_steps(),
                    "encoding_params": self.get_encoding_params()
                } if self.is_snn() else None,
                "learning_rate": self.get_learning_rate(),
                "learning_rate_decay": self.get_learning_rate_decay(),
                "milestones": self.get_milestones()
            },
            "aggregator": self.get_aggregator_info(),
            "pre_aggregators": self.get_preaggregators(),
            "honest_clients": {
                "momentum": self.get_honest_clients_momentum(),
                "weight_decay": self.get_honest_clients_weight_decay(),
                "batch_size": self.get_honest_clients_batch_size()
            },
            "attack": self.get_attack_info(),
            "evaluation_and_results": {
                "evaluation_delta": self.get_evaluation_delta(),
                "batch_size_evaluation": self.get_batch_size_evaluation(),
                "evaluate_on_test": self.get_evaluate_on_test(),
                "store_per_client_metrics": self.get_store_per_client_metrics(),
                "store_models": self.get_store_models(),
                "data_folder": self.get_data_folder(),
                "results_directory": self.get_results_directory()
            }
        }

    # ----------------------------------------------------------------------
    #  Benchmark Config
    # ----------------------------------------------------------------------

    def get_device(self):
        default = "cpu"
        path = ["benchmark_config", "device"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_training_seed(self):
        default = 0
        path = ["benchmark_config", "training_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_training_seeds(self):
        default = 1
        path = ["benchmark_config", "nb_training_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_workers(self):
        default = 1
        path = ["benchmark_config", "nb_workers"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_nb_honest_clients(self):
        default = 0
        path = ["benchmark_config", "nb_honest_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_f(self):
        default = 0
        path = ["benchmark_config", "f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_tolerated_f(self):
        default = self.get_f()
        path = ["benchmark_config", "tolerated_f"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_set_honest_clients_as_clients(self):
        default = False
        path = ["benchmark_config", "set_honest_clients_as_clients"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_size_train_set(self):
        default = 0.8
        path = ["benchmark_config", "size_train_set"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_distribution_seed(self):
        default = 0
        path = ["benchmark_config", "data_distribution_seed"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_data_distribution_seeds(self):
        default = 1
        path = ["benchmark_config", "nb_data_distribution_seeds"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_data_distribution(self):
        default = {
                "name": "iid",
                "distribution_parameter": 1.0
        }
        path = ["benchmark_config", "data_distribution"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_name_data_distribution(self):
        default = "iid"
        path = ["benchmark_config", "data_distribution", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_parameter_data_distribution(self):
        default = 1.0
        path = ["benchmark_config", "data_distribution", "distribution_parameter"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm(self):
        default = {
            "name": "DSGD",
            "parameters": {}
        }
        path = ["benchmark_config", "training_algorithm"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm_name(self):
        default = "DSGD"
        path = ["benchmark_config", "training_algorithm", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_training_algorithm_parameters(self):
        default = {}
        path = ["benchmark_config", "training_algorithm", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_steps(self):
        default = 1000
        path = ["benchmark_config", "nb_steps"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_byzantine_removal_step(self):
        """Training step at which Byzantine clients stop being aggregated (irreversibility test). None disables removal."""
        default = None
        path = ["benchmark_config", "byzantine_removal_step"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Model
    # ----------------------------------------------------------------------
    def get_model_name(self):
        default = "cnn_mnist"
        path = ["model", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_dataset_name(self):
        default = "mnist"
        path = ["model", "dataset_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_nb_labels(self):
        default = 10
        path = ["model", "nb_labels"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_loss_name(self):
        path = ["model", "loss"]
        read = self._read_object(path)
        if read is not None:
            return read
            
        if self.is_snn():
            enc_type = self.get_encoding_type().lower()
            if enc_type == "latency":
                return "ce_temporal_loss"
            else:
                return "ce_rate_loss"
                
        return "NLLLoss"
    
    def get_optimizer_name(self):
        default = "SGD"
        path = ["model", "optimizer_name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate(self):
        default = 0.1
        path = ["model", "learning_rate"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_learning_rate_decay(self):
        default = 1.0
        path = ["model", "learning_rate_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_milestones(self):
        default = []
        path = ["model", "milestones"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Aggregator
    # ----------------------------------------------------------------------
    def get_aggregator_info(self):
        default = {"name": "Average", "parameters": {}}
        path = ["aggregator"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_aggregator_name(self):
        default = "average"
        path = ["aggregator", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_aggregator_parameters(self):
        default = {}
        path = ["aggregator", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Pre-Aggregators
    # ----------------------------------------------------------------------
    def get_preaggregators(self):
        default = []
        path = ["pre_aggregators"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Honest Nodes
    # ----------------------------------------------------------------------
    def get_honest_clients_momentum(self):
        default = 0.9
        path = ["honest_clients", "momentum"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_honest_clients_weight_decay(self):
        default = 1e-4
        path = ["honest_clients", "weight_decay"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_honest_clients_batch_size(self):
        default = 32
        path = ["honest_clients", "batch_size"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_honest_clients_gradient_clip_val(self):
        default = 0.0
        path = ["honest_clients", "gradient_clip_val"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Attack
    # ----------------------------------------------------------------------

    def get_attack_info(self):
        default = {"name": "NoAttack", "parameters": {}}
        path = ["attack"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_attack_name(self):
        default = "NoAttack"
        path = ["attack", "name"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_attack_parameters(self):
        default = {}
        path = ["attack", "parameters"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  Evaluation and Results Accessors
    # ----------------------------------------------------------------------
    def get_evaluation_delta(self):
        default = 50
        path = ["evaluation_and_results", "evaluation_delta"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_batch_size_evaluation(self):
        default = 128
        path = ["evaluation_and_results", "batch_size_evaluation"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_evaluate_on_test(self):
        default = True
        path = ["evaluation_and_results", "evaluate_on_test"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)
    
    def get_store_per_client_metrics(self):
        default = True
        path = ["evaluation_and_results", "store_per_client_metrics"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_store_models(self):
        default = False
        path = ["evaluation_and_results", "store_models"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_data_folder(self):
        default = "./data"
        path = ["evaluation_and_results", "data_folder"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    def get_results_directory(self):
        default = "./results"
        path = ["evaluation_and_results", "results_directory"]
        read = self._read_object(path)
        return self._parameter_to_use(default, read)

    # ----------------------------------------------------------------------
    #  SNN Properties
    # ----------------------------------------------------------------------

    def is_snn(self):
        """Check if model is a Spiking Neural Network."""
        val = self._read_object(["model", "is_snn"])
        if val is not None:
            return bool(val)
        model_name = self.get_model_name()
        return "snn" in model_name.lower()

    def get_encoding_type(self):
        """Get SNN encoding type (constant, rate, latency)."""
        val = self._read_object(["model", "encoding", "type"])
        if val is not None:
            return val
        val = self._read_object(["model", "encoding_type"])
        return self._parameter_to_use("constant", val)

    def get_time_steps(self):
        """Get SNN time steps."""
        val = self._read_object(["model", "encoding", "time_steps"])
        if val is not None:
            return val
        val = self._read_object(["model", "time_steps"])
        return self._parameter_to_use(25, val)



    def get_encoding_params(self):
        """Get SNN encoding params."""
        val = self._read_object(["model", "encoding", "encoding_params"])
        if val is not None:
            return val
        val = self._read_object(["model", "encoding_params"])
        return self._parameter_to_use({}, val)

    def get_model_params(self):
        """Get SNN custom model params."""
        val = self._read_object(["model", "model_params"])
        return self._parameter_to_use({}, val)

    def get_loss_params(self):
        """Get SNN custom loss params."""
        val = self._read_object(["model", "loss_params"])
        return self._parameter_to_use({}, val)

    def get_accuracy_name(self):
        """Get SNN accuracy metric name."""
        path = ["model", "accuracy_name"]
        read = self._read_object(path)
        if read is not None:
            return read
            
        if self.is_snn():
            enc_type = self.get_encoding_type().lower()
            if enc_type == "latency":
                return "accuracy_temporal"
            else:
                return "accuracy_rate"
                
        return None

    def _validate_snn_params(self):
        """Warn user if SNN parameters (encoding, loss, accuracy) are mismatched.
        TODO: Voir si c'est pertinent avec Geovani"""
        if not self.is_snn():
            return
            
        enc_type = self.get_encoding_type().lower()
        loss_name = self.get_loss_name()
        acc_name = self.get_accuracy_name()
        
        warnings = []
        if enc_type == "latency":
            if "temporal" not in loss_name.lower():
                warnings.append(f"Mismatch: encoding is '{enc_type}' but loss is '{loss_name}'. Expected a temporal loss (e.g. ce_temporal_loss).")
            if acc_name != "accuracy_temporal":
                warnings.append(f"Mismatch: encoding is '{enc_type}' but accuracy is '{acc_name}'. Expected accuracy_temporal.")
                
        for warn in warnings:
            print(f"WARNING: [SNN Config Validation] {warn}")

    def get_shared_dataset_cache(self):
        """Get shared dataset cache containing pre-loaded shared memory tensors."""
        return self.data.get("shared_dataset_cache", None)