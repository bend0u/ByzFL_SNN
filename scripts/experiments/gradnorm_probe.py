"""
Offline gradient-norm probe for the honest cnn_mnist under the real DSGD loop.

Goal: characterize the distribution of the RAW per-client gradient norm (the
quantity a client-side raw-gradient clip sees, BEFORE momentum) so we can design
a clipping threshold WITHOUT training an SNN and reading its max norm.

Faithful to the benchmark: 10 honest clients share one global model; each step
every client does one backward pass on its own shard's batch on the CURRENT
global model, we record its raw grad norm, then apply client-side momentum
(beta=0.9), average the momentum vectors (f=0, plain mean = no attack), and take
one SGD step (lr 0.15, wd 1e-4) on the global model. gamma_similarity_niid
partition, exactly as in configs/activation_clip/*.

Records, per (gamma, step, client):
  - global raw grad L2 norm
  - per-layer raw grad L2 norm (c1,c2,f1,f2; weight+bias grouped)
  - per-layer weight L2 norm ||W||        -> for AGC ratio ||g||/||W||
And aggregates per-neuron (row-norm) grad distributions per layer.

Outputs npz + a few PNGs under results/activation_clip/gradnorm_probe/.
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from byzfl.fed_framework.models import cnn_mnist
from byzfl.fed_framework.data_distributor import DataDistributor

OUT = os.path.join(REPO, "results", "activation_clip", "gradnorm_probe")
os.makedirs(OUT, exist_ok=True)

LAYERS = ["_c1", "_c2", "_f1", "_f2"]


def flat_grad(model):
    return torch.cat([p.grad.detach().view(-1) for p in model.parameters()])


def layer_grad_norms(model):
    d = {}
    for name in LAYERS:
        mod = getattr(model, name)
        g = torch.cat([mod.weight.grad.view(-1), mod.bias.grad.view(-1)])
        d[name] = torch.linalg.norm(g).item()
    return d


def layer_weight_norms(model):
    d = {}
    for name in LAYERS:
        mod = getattr(model, name)
        w = torch.cat([mod.weight.view(-1), mod.bias.view(-1)])
        d[name] = torch.linalg.norm(w).item()
    return d


def per_neuron_norms(model):
    """Row-wise (output-unit) grad norms of each layer's weight matrix."""
    d = {}
    for name in LAYERS:
        w = getattr(model, name).weight.grad.detach()
        rows = w.reshape(w.shape[0], -1)
        d[name] = torch.linalg.norm(rows, dim=1).cpu().numpy()
    return d


def run_gamma(gamma, nb_clients, nb_steps, lr, mom, wd, bs, device, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    import random; random.seed(seed)

    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,))])
    full = datasets.MNIST(root=os.path.join(REPO, "data"), train=True,
                          download=False, transform=tfm)
    base_loader = DataLoader(full, batch_size=bs, shuffle=True)

    dist = DataDistributor({
        "data_distribution_name": "gamma_similarity_niid",
        "distribution_parameter": float(gamma),
        "nb_honest": nb_clients,
        "data_loader": base_loader,
        "batch_size": bs,
    })
    loaders = dist.split_data()
    iters = [iter(dl) for dl in loaders]

    model = cnn_mnist().to(device)
    # one shared global param vector; client momentum buffers
    mom_buf = [torch.zeros_like(torch.cat([p.view(-1) for p in model.parameters()]),
                                device=device) for _ in range(nb_clients)]

    global_norms = np.zeros((nb_steps, nb_clients), np.float32)
    layer_norms = {L: np.zeros((nb_steps, nb_clients), np.float32) for L in LAYERS}
    wnorms = {L: np.zeros((nb_steps,), np.float32) for L in LAYERS}
    agc_ratio = {L: np.zeros((nb_steps, nb_clients), np.float32) for L in LAYERS}
    neuron_accum = {L: [] for L in LAYERS}  # sampled

    params = list(model.parameters())
    shapes = [p.shape for p in params]
    numels = [p.numel() for p in params]

    def set_flat(vec):
        off = 0
        with torch.no_grad():
            for p, n in zip(params, numels):
                p.copy_(vec[off:off + n].view(p.shape)); off += n

    def get_flat():
        return torch.cat([p.detach().view(-1) for p in params])

    for step in range(nb_steps):
        wn = layer_weight_norms(model)
        for L in LAYERS:
            wnorms[L][step] = wn[L]
        agg = torch.zeros_like(mom_buf[0])
        for c in range(nb_clients):
            try:
                x, y = next(iters[c])
            except StopIteration:
                iters[c] = iter(loaders[c]); x, y = next(iters[c])
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            out = model(x)
            loss = F.nll_loss(out, y)
            loss.backward()
            # weight decay folded into raw grad (as in SGD wd)
            g = flat_grad(model)
            global_norms[step, c] = torch.linalg.norm(g).item()
            ln = layer_grad_norms(model)
            for L in LAYERS:
                layer_norms[L][step, c] = ln[L]
                agc_ratio[L][step, c] = ln[L] / (wn[L] + 1e-12)
            if step in (0, 50, 150, 300, 499):
                pn = per_neuron_norms(model)
                for L in LAYERS:
                    neuron_accum[L].append(pn[L])
            # client-side momentum (raw grad feeds accumulator)
            mom_buf[c].mul_(mom).add_(g, alpha=1 - mom)
            agg += mom_buf[c]
        agg /= nb_clients
        # SGD step with weight decay on global model
        flat_w = get_flat()
        flat_w = flat_w - lr * (agg + wd * flat_w)
        set_flat(flat_w)

    neuron_cat = {L: (np.concatenate(neuron_accum[L]) if neuron_accum[L] else np.array([]))
                  for L in LAYERS}
    return dict(global_norms=global_norms, layer_norms=layer_norms,
                wnorms=wnorms, agc_ratio=agc_ratio, neuron=neuron_cat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[1.0, 0.33, 0.0])
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--mom", type=float, default=0.9)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    results = {}
    for g in args.gammas:
        t0 = time.time()
        results[g] = run_gamma(g, args.clients, args.steps, args.lr, args.mom,
                               args.wd, args.bs, args.device, args.seed)
        gn = results[g]["global_norms"]
        pk = np.percentile(gn, [50, 80, 90, 95, 99, 100])
        print(f"[gamma={g}] {time.time()-t0:.0f}s  raw global grad norm "
              f"p50={pk[0]:.2f} p80={pk[1]:.2f} p90={pk[2]:.2f} "
              f"p95={pk[3]:.2f} p99={pk[4]:.2f} max={pk[5]:.2f}", flush=True)

    np.savez(os.path.join(OUT, "gradnorm_probe.npz"),
             **{f"g{g}_global": results[g]["global_norms"] for g in results},
             **{f"g{g}_{L}": results[g]["layer_norms"][L] for g in results for L in LAYERS},
             **{f"g{g}_w{L}": results[g]["wnorms"][L] for g in results for L in LAYERS},
             **{f"g{g}_agc{L}": results[g]["agc_ratio"][L] for g in results for L in LAYERS},
             **{f"g{g}_neuron{L}": results[g]["neuron"][L] for g in results for L in LAYERS})
    print("saved", os.path.join(OUT, "gradnorm_probe.npz"))


if __name__ == "__main__":
    main()
