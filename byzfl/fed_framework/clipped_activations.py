import torch


class ClippedReLU_STE(torch.autograd.Function):
    """Forward: hard clamp to [0, clip_val]. Backward: straight-through estimator
    on the upper saturation region -- gradient stays 1 for all x > 0, it never
    collapses to 0 above clip_val (unlike plain hardtanh)."""

    @staticmethod
    def forward(ctx, input_, clip_val):
        ctx.save_for_backward(input_)
        return input_.clamp(min=0.0, max=clip_val)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * (input_ > 0).float()
        return grad, None


def clipped_relu_ste(clip_val=1.0):
    def inner(x):
        return ClippedReLU_STE.apply(x, clip_val)
    return inner


class ClippedReLU_Ramp(torch.autograd.Function):
    """Forward: hard clamp to [0, clip_val]. Backward: gradient is 1 on (0, clip_val],
    then decays linearly to 0 between clip_val and ramp_ratio*clip_val, then 0 beyond
    that. Preserves the distinction between "slightly over the clip" and "way over
    the clip" instead of treating both identically (as a hard STE cutoff would)."""

    @staticmethod
    def forward(ctx, input_, clip_val, ramp_ratio):
        ctx.save_for_backward(input_)
        ctx.clip_val = clip_val
        ctx.ramp_ratio = ramp_ratio
        return input_.clamp(min=0.0, max=clip_val)

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        C = ctx.clip_val
        rC = ctx.ramp_ratio * C
        grad_input = grad_output.clone()

        mask = torch.zeros_like(input_)
        # slope 1 region: 0 < x <= C
        mask = torch.where((input_ > 0) & (input_ <= C), torch.ones_like(input_), mask)
        # linear ramp-down region: C < x < rC
        in_ramp = (input_ > C) & (input_ < rC)
        ramp_val = (rC - input_) / (rC - C)
        mask = torch.where(in_ramp, ramp_val, mask)
        # x <= 0 or x >= rC -> 0 (already the default)

        grad = grad_input * mask
        return grad, None, None


def clipped_relu_ramp(clip_val=1.0, ramp_ratio=2.0):
    def inner(x):
        return ClippedReLU_Ramp.apply(x, clip_val, ramp_ratio)
    return inner


class AdaptiveQuantileClip(torch.autograd.Function):
    """Forward: clamp to [0, C] where C is the tau-quantile of this layer's own
    activation coordinates for the current forward pass (detached -- treated as a
    constant for backward purposes), so the threshold adapts to the data instead of
    being a fixed hand-picked constant. Backward: either the plain clamp derivative
    (0 above C, same failure mode as hardtanh) or a straight-through estimator
    (gradient stays 1 above C), selected via `ste`."""

    @staticmethod
    def forward(ctx, input_, tau, ste):
        C = torch.quantile(input_.detach().flatten(), tau)
        ctx.save_for_backward(input_)
        ctx.clip_val = C
        ctx.ste = ste
        return input_.clamp(min=0.0, max=C.item())

    @staticmethod
    def backward(ctx, grad_output):
        (input_,) = ctx.saved_tensors
        C = ctx.clip_val
        grad_input = grad_output.clone()
        if ctx.ste:
            mask = (input_ > 0).float()
        else:
            mask = ((input_ > 0) & (input_ <= C)).float()
        grad = grad_input * mask
        return grad, None, None


def adaptive_qclip(tau=0.8, ste=False):
    def inner(x):
        return AdaptiveQuantileClip.apply(x, tau, ste)
    return inner
