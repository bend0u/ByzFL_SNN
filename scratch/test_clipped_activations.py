"""
Unit smoke test for byzfl/fed_framework/clipped_activations.py.

These backward passes are INTENTIONALLY not the true derivative of clamp (that's
the whole point -- STE/ramp/adaptive exist to avoid the true derivative's dead
zone), so torch.autograd.gradcheck will NOT pass. Instead we assert forward
values and backward gradients against hand-derived expected masks at
representative points.
"""
import torch
import sys
import os
import importlib.util

# Load clipped_activations.py directly by file path rather than via
# `from byzfl.fed_framework.clipped_activations import ...`, because that import
# chain goes through byzfl/__init__.py -> fed_framework/__init__.py -> models.py,
# which unconditionally imports snntorch. clipped_activations.py itself only
# depends on torch, so this sidesteps needing snntorch installed just to run
# this unit test.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "byzfl", "fed_framework", "clipped_activations.py",
)
_spec = importlib.util.spec_from_file_location("clipped_activations", _MODULE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

ClippedReLU_STE = _mod.ClippedReLU_STE
ClippedReLU_Ramp = _mod.ClippedReLU_Ramp
AdaptiveQuantileClip = _mod.AdaptiveQuantileClip
clipped_relu_ste = _mod.clipped_relu_ste
clipped_relu_ramp = _mod.clipped_relu_ramp
adaptive_qclip = _mod.adaptive_qclip

TOL = 1e-6


def check(name, actual, expected):
    ok = torch.allclose(actual, expected, atol=TOL)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}")
    if not ok:
        print(f"    actual:   {actual.tolist()}")
        print(f"    expected: {expected.tolist()}")
    return ok


def main():
    all_ok = True

    # ---- ClippedReLU_STE: C=1 ----
    x = torch.tensor([-2.0, -0.5, 0.0, 0.3, 1.0, 1.5, 3.0], requires_grad=True)
    C = 1.0
    y = ClippedReLU_STE.apply(x, C)
    expected_fwd = x.detach().clamp(0.0, C)
    all_ok &= check("STE forward == clamp(x,0,C)", y.detach(), expected_fwd)

    y.sum().backward()
    # backward mask: 1 for x>0, else 0 (never collapses above C)
    expected_grad = (x.detach() > 0).float()
    all_ok &= check("STE backward == (x>0)", x.grad, expected_grad)

    # ---- ClippedReLU_Ramp: C=1, r=2 -> ramp end at 2 ----
    x2 = torch.tensor([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5], requires_grad=True)
    C, r = 1.0, 2.0
    y2 = ClippedReLU_Ramp.apply(x2, C, r)
    expected_fwd2 = x2.detach().clamp(0.0, C)
    all_ok &= check("Ramp forward == clamp(x,0,C)", y2.detach(), expected_fwd2)

    y2.sum().backward()
    # hand-derived: x=-1->0, x=0->0 (boundary, not >0), x=0.5->1, x=1.0->1,
    # x=1.5-> (2-1.5)/(2-1)=0.5, x=2.0-> (2-2)/1=0.0, x=2.5(>=rC)->0
    expected_grad2 = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 0.0])
    all_ok &= check("Ramp backward piecewise mask", x2.grad, expected_grad2)

    # ---- AdaptiveQuantileClip: plain backward ----
    x3 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    tau = 0.8
    C3 = torch.quantile(x3.detach(), tau)
    y3 = AdaptiveQuantileClip.apply(x3, tau, False)
    expected_fwd3 = x3.detach().clamp(0.0, C3.item())
    all_ok &= check("AdaptiveQuantileClip forward == clamp(x,0,quantile)", y3.detach(), expected_fwd3)

    y3.sum().backward()
    expected_grad3_plain = ((x3.detach() > 0) & (x3.detach() <= C3)).float()
    all_ok &= check("AdaptiveQuantileClip plain backward == (0<x<=C)", x3.grad, expected_grad3_plain)

    # ---- AdaptiveQuantileClip: STE backward ----
    x4 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    C4 = torch.quantile(x4.detach(), tau)
    y4 = AdaptiveQuantileClip.apply(x4, tau, True)
    y4.sum().backward()
    expected_grad4_ste = (x4.detach() > 0).float()
    all_ok &= check("AdaptiveQuantileClip STE backward == (x>0)", x4.grad, expected_grad4_ste)

    # ---- Functional wrapper closures behave the same as direct .apply ----
    x5 = torch.tensor([0.5, 1.5], requires_grad=True)
    y5 = clipped_relu_ste(1.0)(x5)
    all_ok &= check("clipped_relu_ste wrapper forward", y5.detach(), x5.detach().clamp(0.0, 1.0))

    x6 = torch.tensor([0.5, 1.5], requires_grad=True)
    y6 = clipped_relu_ramp(1.0, 2.0)(x6)
    all_ok &= check("clipped_relu_ramp wrapper forward", y6.detach(), x6.detach().clamp(0.0, 1.0))

    x7 = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y7 = adaptive_qclip(0.8, False)(x7)
    all_ok &= check("adaptive_qclip wrapper forward", y7.detach(), x7.detach().clamp(0.0, C3.item()))

    print()
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
