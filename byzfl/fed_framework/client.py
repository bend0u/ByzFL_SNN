import math
import torch
import numpy as np
from collections import deque

from byzfl.fed_framework import ModelBaseInterface
from byzfl.utils.conversion import flatten_dict
from byzfl.utils.snn_loss import SNN_LOSS_REGISTRY, create_snn_loss
import snntorch as snn
import snntorch.functional as sf

class Client(ModelBaseInterface):

    def __init__(self, params):
        # Check for correct types and values in params
        if not isinstance(params, dict):
            raise TypeError(f"'params' must be of type dict, but got {type(params).__name__}")
        if not isinstance(params["loss_name"], str):
            raise TypeError(f"'loss_name' must be of type str, but got {type(params['loss_name']).__name__}")
        if not isinstance(params["LabelFlipping"], bool):
            raise TypeError(f"'LabelFlipping' must be of type bool, but got {type(params['LabelFlipping']).__name__}")
        if not isinstance(params["nb_labels"], int) or not params["nb_labels"] > 1:
            raise ValueError(f"'nb_labels' must be an integer greater than 1")
        if not isinstance(params["momentum"], float) or not 0 <= params["momentum"] < 1:
            raise ValueError(f"'momentum' must be a float in the range [0, 1)")
        if not isinstance(params["training_dataloader"], torch.utils.data.DataLoader):
            raise TypeError(f"'training_dataloader' must be a DataLoader, but got {type(params['training_dataloader']).__name__}")

        # Initialize Client instance
        super().__init__({
            # Required parameters
            "model_name": params["model_name"],
            "device": params["device"],
            # Optional parameters
            "learning_rate": params.get("learning_rate", None),
            "weight_decay": params.get("weight_decay", None),
            "milestones": params.get("milestones", None),
            "learning_rate_decay": params.get("learning_rate_decay", None),
            "optimizer_name": params.get("optimizer_name", None),
            "optimizer_params": params.get("optimizer_params", {}),
            "model_params": params.get("model_params", {}),
        })

        loss_name = params["loss_name"]
        loss_params = params.get("loss_params", {})
        #We check if the loss function is one for SNN and if so we use it. Else, we search the loss function in pytorch
        if loss_name in SNN_LOSS_REGISTRY:
            self.criterion = create_snn_loss(loss_name, **loss_params)
        else:
            self.criterion = getattr(torch.nn, loss_name)()

        accuracy_name = params.get("accuracy_name", None)
        if accuracy_name is not None:
            if hasattr(sf, accuracy_name):
                self.accuracy_fn = getattr(sf, accuracy_name)
            else:
                raise ValueError(
                    f"Accuracy function '{accuracy_name}' not found in snntorch.functional. "
                    f"Please choose a valid one (e.g. 'accuracy_rate', 'accuracy_temporal')"
                )
        else:
            self.accuracy_fn = None
        self.gradient_LF = 0
        self.labelflipping = params["LabelFlipping"]
        self.nb_labels = params["nb_labels"]
        self.momentum = params["momentum"]
        self.momentum_gradient = torch.zeros_like(
            torch.cat(tuple(
                tensor.view(-1) 
                for tensor in self.model.parameters()
            )),
            device=params["device"]
        )
        self.training_dataloader = params["training_dataloader"]
        self.train_iterator = iter(self.training_dataloader)
        self.store_per_client_metrics = params["store_per_client_metrics"]
        self.gradient_clip_val = params.get("gradient_clip_val", 0.0)

        # Adaptive client-side gradient-norm clip: each client clips the L2 norm
        # of the (post-momentum) vector it sends to the server to the
        # grad_clip_quantile-quantile of its OWN last grad_clip_window gradient
        # norms. Windowed (not a global/all-time quantile) because gradient norms
        # are non-stationary -- they shrink as training converges, so a global
        # quantile would stop clipping anything late in training.
        self.grad_clip_quantile = params.get("grad_clip_quantile", 0.0)
        self.grad_clip_window = params.get("grad_clip_window", 100)
        self._grad_norm_history = deque(maxlen=self.grad_clip_window)

        # Same adaptive windowed-quantile mechanism, but applied to the RAW
        # gradient BEFORE it enters the momentum accumulator (rather than to the
        # post-momentum vector on its way out). This is the adaptive counterpart
        # of the fixed `gradient_clip_val`, which also clips the raw gradient --
        # so the two are directly comparable, and bounding what feeds the
        # accumulator prevents unbounded growth (and the resulting overflow/NaN).
        self.raw_grad_clip_quantile = params.get("raw_grad_clip_quantile", 0.0)
        self.raw_grad_clip_window = params.get("raw_grad_clip_window", 100)
        self._raw_grad_norm_history = deque(maxlen=self.raw_grad_clip_window)

        self.loss_list = list()
        self.train_acc_list = list()

        # Firing-rate instrumentation (EXP2): mean spike rate across all LIF
        # layers on the client's own forward pass, one scalar per compute_gradients() call.
        # No-op (get_last_firing_rate() returns None) for non-spiking models.
        self._spike_batch_means = []
        self._last_firing_rate = None
        # Per-layer breakdown (threshold sweep): same spikes, bucketed by layer
        # name instead of pooled into one flat list.
        self._layer_spike_means = {}
        self._last_layer_firing_rates = {}
        spike_layers = [m for m in self.model.modules() if isinstance(m, snn.Leaky)]
        self._layer_names = [f"lif{i + 1}" for i in range(len(spike_layers))]
        for layer_name, layer in zip(self._layer_names, spike_layers):
            layer.register_forward_hook(self._make_spike_hook(layer_name))

    def _make_spike_hook(self, layer_name):
        def hook(module, layer_input, layer_output):
            spikes = layer_output[0]
            mean = spikes.mean().item()
            self._spike_batch_means.append(mean)
            self._layer_spike_means.setdefault(layer_name, []).append(mean)
        return hook

    def get_last_firing_rate(self):
        """Mean fraction of neurons spiking, averaged over all LIF layers and
        timesteps of the most recent forward pass. None for non-spiking models."""
        return self._last_firing_rate

    def get_last_layer_firing_rates(self):
        """Dict {layer_name: mean firing rate} for the most recent forward
        pass, one entry per LIF layer (averaged over timesteps). Empty dict
        for non-spiking models."""
        return self._last_layer_firing_rates

    def _sample_train_batch(self):
        """
        Description
        -----------
        Retrieves the next batch of data from the training dataloader. If the 
        end of the dataset is reached, the dataloader is reinitialized to start 
        from the beginning.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding target labels for the current batch.
        """
        try:
            return next(self.train_iterator)
        except StopIteration:
            self.train_iterator = iter(self.training_dataloader)
            return next(self.train_iterator)

    def compute_gradients(self):
        """
        Description
        -----------
        Computes the gradients of the local model's loss function for the 
        current training batch. If the `LabelFlipping` attack is enabled, 
        gradients for flipped targets are computed and stored separately. 
        Additionally, the training loss and accuracy for the batch are 
        computed and recorded.
        """
        inputs, targets = self._sample_train_batch()
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        if self.labelflipping:
            self.model.eval()
            targets_flipped = targets.sub(self.nb_labels - 1).mul(-1)
            self._backward_pass(inputs, targets_flipped)
            self.gradient_LF = self.get_dict_gradients()
            self.model.train()

        train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)

        if self.store_per_client_metrics:
            self.loss_list.append(train_loss_value)

        return train_loss_value

    def _backward_pass(self, inputs, targets, train_acc=False):
        """
        Description
        -----------
        Performs a backward pass through the model to compute gradients for 
        the given inputs and targets. Optionally computes training accuracy 
        for the batch.

        Parameters
        ----------
        inputs : torch.Tensor
            The input data for the batch.
        targets : torch.Tensor
            The target labels for the batch.
        train_acc : bool, optional
            If True, computes and stores the training accuracy for the batch. 
            Default is False.

        Returns
        -------
        float
            The loss value for the current batch.
        """
        self.model.zero_grad()
        self._spike_batch_means = []
        self._layer_spike_means = {}
        outputs = self.model(inputs)
        self._last_firing_rate = (
            float(np.mean(self._spike_batch_means)) if self._spike_batch_means else None
        )
        self._last_layer_firing_rates = {
            layer_name: float(np.mean(vals))
            for layer_name, vals in self._layer_spike_means.items()
        }
        loss = self.criterion(outputs, targets)
        loss_value = loss.item()
        loss.backward()

        # Apply gradient clipping if configured
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)

        if train_acc:
            # Compute and store train accuracy
            is_snn = isinstance(outputs, tuple)
            if is_snn:
                fn = self.accuracy_fn if self.accuracy_fn is not None else sf.accuracy_rate
                acc = fn(outputs[0], targets)
            else:
                _, predicted = torch.max(outputs.data, 1)
                total = targets.size(0)
                correct = (predicted == targets).sum().item()
                acc = correct / total
            self.train_acc_list.append(acc)

        return loss_value
    
    def compute_model_update(self, num_rounds):
        """
        Description
        -----------
        Executes multiple rounds of training updates on the model. For each round,
        it samples a batch of training data, performs a backward pass to compute
        gradients, and updates the model parameters. Optionally logs training loss
        and accuracy.

        Parameters
        ----------
        num_rounds : int
            The number of training iterations to perform. Each iteration includes
            sampling a batch, computing the loss and gradients, and updating the model.

        Returns
        -------
        float
            The mean loss across all training rounds.
        """

        losses = np.zeros((num_rounds))
        for i in range(num_rounds):
            inputs, targets = self._sample_train_batch()
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            train_loss_value = self._backward_pass(inputs, targets, train_acc=self.store_per_client_metrics)
            losses[i] = train_loss_value
            self.optimizer.step()

            if self.store_per_client_metrics:
                self.loss_list.append(train_loss_value)

        return losses.mean()
            

    def get_flat_flipped_gradients(self):
        """
        Description
        -----------
        Retrieves the gradients computed using flipped targets as a flat array.

        Returns
        -------
        numpy.ndarray or torch.Tensor
            A flat array containing the gradients for the model parameters 
            when trained with flipped targets.
        """
        return flatten_dict(self.gradient_LF)

    def get_flat_gradients_with_momentum(self):
        """
        Description
        -----------
        Computes the gradients with momentum applied and returns them as a flat
        array.

        Two OPTIONAL and INDEPENDENT adaptive clips can be applied, differing only
        in WHERE they sit relative to the momentum accumulator. Both use the same
        mechanism (`_clip_to_windowed_quantile`): clip the L2 norm to the
        q-quantile of this client's own last W norms. Both are purely client-side.

          raw_grad_clip_quantile  -- clips the RAW gradient BEFORE it enters the
            accumulator. Because it bounds what feeds the recursion
            (v <- beta*v + (1-beta)*g), it also prevents v from growing without
            bound. This is the same position as the fixed `gradient_clip_val`
            (applied in `_backward_pass`), so it is that clip's adaptive
            counterpart.

          grad_clip_quantile      -- clips the POST-momentum vector on its way to
            the server. The internal momentum buffer is never rescaled: only the
            returned copy is bounded, so what the aggregator sees is bounded while
            the client's own momentum dynamics stay untouched. Note this canNOT
            prevent the accumulator itself from diverging.

        Enabling both is allowed; they are applied in pipeline order (raw first,
        then post-momentum) and track separate norm histories.

        Returns
        -------
        torch.Tensor
            A flat array containing the gradients with momentum applied.
        """
        raw_gradient = self.get_flat_gradients()

        # Optionally clip the RAW gradient before it enters the momentum
        # accumulator (see raw_grad_clip_quantile in __init__).
        if self.raw_grad_clip_quantile > 0:
            raw_gradient = self._clip_to_windowed_quantile(
                raw_gradient,
                self._raw_grad_norm_history,
                self.raw_grad_clip_quantile,
                self.raw_grad_clip_window,
            )

        self.momentum_gradient.mul_(self.momentum)
        self.momentum_gradient.add_(raw_gradient, alpha=1 - self.momentum)

        if self.grad_clip_quantile <= 0:
            return self.momentum_gradient

        # Clip the POST-momentum vector on its way to the server.
        return self._clip_to_windowed_quantile(
            self.momentum_gradient,
            self._grad_norm_history,
            self.grad_clip_quantile,
            self.grad_clip_window,
        )

    def _clip_to_windowed_quantile(self, vector, norm_history, quantile, window):
        """
        Description
        -----------
        Rescales `vector` so its L2 norm does not exceed the `quantile`-quantile
        of the last `window` norms recorded in `norm_history` (a sliding window,
        since gradient norms are non-stationary and shrink as training converges).

        The pre-clip norm is what gets recorded, so the quantile reflects the true
        (unclipped) distribution. Returns a rescaled copy; `vector` is never
        modified in place. During warmup (insufficient history) the vector is
        returned unchanged.

        Non-finite norms are skipped rather than recorded: a single NaN/Inf would
        otherwise poison every subsequent quantile (making the threshold NaN, which
        silently disables clipping since all NaN comparisons are False).

        Returns
        -------
        torch.Tensor
            The (possibly rescaled) vector.
        """
        current_norm = torch.linalg.norm(vector).item()
        if not math.isfinite(current_norm):
            return vector

        norm_history.append(current_norm)

        min_history = max(2, window // 2)
        if len(norm_history) < min_history:
            # Not enough history yet to estimate a stable quantile -- skip clipping.
            return vector

        threshold = np.quantile(norm_history, quantile)
        if current_norm > threshold and current_norm > 0:
            return vector * (threshold / current_norm)
        return vector

    def get_loss_list(self):
        """
        Description
        -----------
        Retrieves the list of training losses recorded over the course of 
        training.

        Returns
        -------
        list
            A list of float values representing the training losses for each 
            batch.
        """
        return self.loss_list

    def get_train_accuracy(self):
        """
        Description
        -----------
        Retrieves the training accuracy for each batch processed during 
        training.

        Returns
        -------
        list
            A list of float values representing the training accuracy for each 
            batch.
        """
        return self.train_acc_list

    def set_model_state(self, state_dict):
        """
        Description
        -----------
        Updates the state of the model with the provided state dictionary. 
        This method is used to load a saved model state or update 
        the global model in a federated learning context.
        Typically, this method can be used to synchronize clients with the global model.

        Parameters
        ----------
        state_dict : dict
            The state dictionary containing model parameters and buffers.

        Raises
        ------
        TypeError
            If `state_dict` is not a dictionary.
        """
        if not isinstance(state_dict, dict):
            raise TypeError(f"'state_dict' must be of type dict, but got {type(state_dict).__name__}")
        self.model.load_state_dict(state_dict)