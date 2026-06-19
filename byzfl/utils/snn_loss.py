import torch.nn as nn
import snntorch.functional as sf

# Registry mapping snntorch loss function names to the index of the model output
# they require. SNN models return (spk_rec, mem_rec) tuples:
#   index 0 = spike records
#   index 1 = membrane potential records
SNN_LOSS_REGISTRY = {
    "ce_rate_loss":         0,  # spike-based
    "ce_count_loss":        0,  # spike-based
    "mse_count_loss":       0,  # spike-based
    "ce_max_membrane_loss": 1,  # membrane-based
    "mse_membrane_loss":    1,  # membrane-based
    "ce_temporal_loss":     0,  # temporal spike-based
    "mse_temporal_loss":    0,  # temporal spike-based
}


class SNNLoss(nn.Module):
    """
    Wrapper around snntorch loss functions that transparently handles
    both single-tensor and tuple model outputs.

    Description
    -----------
    SNN models typically return a tuple ``(spk_rec, mem_rec)`` from their
    forward pass. Different loss functions require different parts of this
    output (spikes vs membrane potentials). This wrapper automatically
    extracts the correct tensor based on the ``input_index`` before
    delegating to the underlying snntorch loss function.

    For standard (non-SNN) models that return a single tensor, the wrapper
    passes the tensor through unchanged.

    Parameters
    ----------
    loss_fn : callable
        An instantiated snntorch loss function
        (e.g., ``sf.ce_rate_loss()``).
    input_index : int
        Which element of the model output tuple to use.
        0 for spike records, 1 for membrane potentials.
    """

    def __init__(self, loss_fn, input_index=0):
        super().__init__()
        self.loss_fn = loss_fn
        self.input_index = input_index

    def forward(self, outputs, targets):
        """
        Parameters
        ----------
        outputs : torch.Tensor or tuple of torch.Tensor
            Raw model output. Can be a single tensor (standard models)
            or a tuple ``(spk_rec, mem_rec)`` (SNN models).
        targets : torch.Tensor
            Target class labels.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if isinstance(outputs, tuple):
            tensor = outputs[self.input_index]
        else:
            tensor = outputs
        return self.loss_fn(tensor, targets)


def create_snn_loss(loss_name, **loss_params):
    """
    Factory function that creates an SNNLoss wrapper for a given
    snntorch loss function name.

    Description
    -----------
    Looks up the loss function in ``snntorch.functional`` via ``getattr``,
    instantiates it with optional parameters, and wraps it in an
    ``SNNLoss`` module that handles tuple unpacking automatically.

    Parameters
    ----------
    loss_name : str
        Name of the snntorch loss function (e.g., ``"ce_rate_loss"``).
    **loss_params : dict
        Optional keyword arguments passed to the snntorch loss function
        constructor (e.g., ``correct_rate=1.0``).

    Returns
    -------
    SNNLoss
        A wrapped loss module ready to use as ``criterion(outputs, targets)``.

    Raises
    ------
    ValueError
        If ``loss_name`` is not found in the SNN_LOSS_REGISTRY.
    AttributeError
        If ``loss_name`` does not exist in ``snntorch.functional``.
    """
    if loss_name not in SNN_LOSS_REGISTRY:
        raise ValueError(
            f"SNN loss '{loss_name}' is not registered. "
            f"Available losses: {list(SNN_LOSS_REGISTRY.keys())}"
        )

    input_index = SNN_LOSS_REGISTRY[loss_name]
    loss_fn = getattr(sf, loss_name)(**loss_params)
    return SNNLoss(loss_fn, input_index)
