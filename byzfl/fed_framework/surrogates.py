import torch
from snntorch import surrogate

class Rectangular(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, beta=0.5):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Constant height 1/(2*beta) within the support length 2*beta (-beta, beta)
        in_range = (torch.abs(input_) < ctx.beta).float()
        grad = grad_input * in_range / (2.0 * ctx.beta)
        return grad, None

def box(beta=0.5):
    def inner(x):
        return Rectangular.apply(x, beta)
    return inner

class TriangularNormalized(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, beta=0.5):
        ctx.save_for_backward(input_)
        ctx.beta = beta
        return (input_ > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Peak 1/beta at x=0, linear slope 1/beta^2, zero outside [-beta, beta]
        abs_in = torch.abs(input_)
        in_range = (abs_in < ctx.beta).float()
        tri_grad = (1.0 / ctx.beta) * (1.0 - abs_in / ctx.beta)
        grad = grad_input * tri_grad * in_range
        return grad, None

def tri(beta=0.5):
    def inner(x):
        return TriangularNormalized.apply(x, beta)
    return inner

def get_spike_grad(surrogate_gradient, surrogate_params):
    params = surrogate_params if surrogate_params is not None else {}
    if surrogate_gradient in ["box", "rectangular"]:
        return box(**params)
    elif surrogate_gradient in ["tri", "triangular"]:
        return tri(**params)
    return getattr(surrogate, surrogate_gradient)(**params)
