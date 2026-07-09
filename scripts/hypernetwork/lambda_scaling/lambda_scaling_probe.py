"""
One-shot probe: measure CE scale + gate-gradient stats per base model,
then check H2/H3 λ-scaling hypotheses against empirical λ_opt from sweeps.

Run from project root:
  venv/bin/python scripts/hypernetwork/lambda_scaling_probe.py
"""

import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.prune_train import masked_forward
from scripts.base.train_cifar import CIFARNetBig, CIFARNetLeNet, CIFAR_MEAN, CIFAR_STD
import torchvision
from torchvision.transforms import v2


# Empirical λ_opt from efficiency peaks (crisp-findings F13/F14 + mnist sweep)
EMPIRICAL = {
    "MNIST [1024,1024]": {"lambda_opt": 0.06, "L": 2, "mean_S": 1024, "N_hidden": 2048},
    "CIFAR_big fc":      {"lambda_opt": 0.03, "L": 3, "mean_S": 597.3, "N_hidden": 1792},
    "CIFAR LeNet fc":    {"lambda_opt": 0.06, "L": 2, "mean_S": 96, "N_hidden": 192},
    "wide [2048]":       {"lambda_opt": 0.10, "L": 1, "mean_S": 2048, "N_hidden": 2048},  # iso @2pp
    "deep [512x4]":      {"lambda_opt": 0.80, "L": 4, "mean_S": 512, "N_hidden": 2048},
    "narrow [205,205]":  {"lambda_opt": 0.06, "L": 2, "mean_S": 205, "N_hidden": 410},
}


def measure_mnist(hidden_dims, ckpt_path, device, n_batches=20):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MLP(784, hidden_dims, 10, dropout=0.0).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    _, test_loader = get_mnist_loaders(batch_size=256)
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    gates = [torch.ones(L.weight.shape[0], device=device, requires_grad=True)
             for L in linears[:-1]]

    ce_vals, grad_vals = [], []
    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            ce_orig = F.cross_entropy(model(x), y).item()
        logits = masked_forward(model, gates, x)
        ce = F.cross_entropy(logits, y)
        ce_vals.append(ce.item())
        ce.backward()
        for g in gates:
            grad_vals.extend(g.grad.abs().detach().cpu().tolist())
        for g in gates:
            g.grad = None

    return {
        "ce_mean": sum(ce_vals) / len(ce_vals),
        "grad_mean": sum(grad_vals) / len(grad_vals),
        "grad_median": sorted(grad_vals)[len(grad_vals) // 2],
    }


def measure_cifar_big(device, n_batches=20):
    ckpt = torch.load("experiments/checkpoints/cifar_big.pt", map_location=device)
    model = CIFARNetBig(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tf = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader = DataLoader(test, batch_size=256, shuffle=True)

    gates = [
        torch.ones(model.fc1.weight.shape[0], device=device, requires_grad=True),
        torch.ones(model.fc2.weight.shape[0], device=device, requires_grad=True),
        torch.ones(model.fc3.weight.shape[0], device=device, requires_grad=True),
    ]

    ce_vals, grad_vals = [], []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        h = model.pool(F.relu(model.bn1(model.conv1(x))))
        h = model.pool(F.relu(model.bn2(model.conv2(h))))
        h = model.pool(F.relu(model.bn3(model.conv3(h))))
        h = h.view(h.size(0), -1)
        for linear, gate in [(model.fc1, gates[0]), (model.fc2, gates[1]), (model.fc3, gates[2])]:
            w = linear.weight.detach() * gate.unsqueeze(1)
            b = linear.bias.detach() * gate
            h = F.relu(F.linear(h, w, b))
        ce = F.cross_entropy(model.fc4(h), y)
        ce_vals.append(ce.item())
        ce.backward()
        for g in gates:
            grad_vals.extend(g.grad.abs().detach().cpu().tolist())
        for g in gates:
            g.grad = None

    return {
        "ce_mean": sum(ce_vals) / len(ce_vals),
        "grad_mean": sum(grad_vals) / len(grad_vals),
        "grad_median": sorted(grad_vals)[len(grad_vals) // 2],
    }


def measure_cifar_lenet(device, n_batches=20):
    ckpt = torch.load("experiments/checkpoints/cifar_cnn.pt", map_location=device)
    model = CIFARNetLeNet(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tf = v2.Compose([
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    test = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=tf)
    loader = DataLoader(test, batch_size=256, shuffle=True)

    gates = [
        torch.ones(model.fc1.weight.shape[0], device=device, requires_grad=True),
        torch.ones(model.fc2.weight.shape[0], device=device, requires_grad=True),
    ]

    ce_vals, grad_vals = [], []
    for i, (x, y) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        h = model.pool(F.relu(model.conv1(x)))
        h = model.pool(F.relu(model.conv2(h)))
        h = h.view(h.size(0), -1)
        for linear, gate in [(model.fc1, gates[0]), (model.fc2, gates[1])]:
            w = linear.weight.detach() * gate.unsqueeze(1)
            b = linear.bias.detach() * gate
            h = F.relu(F.linear(h, w, b))
        ce = F.cross_entropy(model.fc3(h), y)
        ce_vals.append(ce.item())
        ce.backward()
        for g in gates:
            grad_vals.extend(g.grad.abs().detach().cpu().tolist())
        for g in gates:
            g.grad = None

    return {
        "ce_mean": sum(ce_vals) / len(ce_vals),
        "grad_mean": sum(grad_vals) / len(grad_vals),
        "grad_median": sorted(grad_vals)[len(grad_vals) // 2],
    }


def main():
    device = torch.device("cpu")
    rows = []

    probes = [
        ("MNIST [1024,1024]", lambda: measure_mnist([1024, 1024], "experiments/checkpoints/mnist_model.pt", device)),
        ("wide [2048]",       lambda: measure_mnist([2048], "experiments/checkpoints/mnist_wide2048.pt", device)),
        ("deep [512x4]",      lambda: measure_mnist([512]*4, "experiments/checkpoints/mnist_deep4x512.pt", device)),
        ("narrow [205,205]",  lambda: measure_mnist([205, 205], "experiments/checkpoints/mnist_narrow205x2.pt", device)),
        ("CIFAR_big fc",      lambda: measure_cifar_big(device)),
        ("CIFAR LeNet fc",    lambda: measure_cifar_lenet(device)),
    ]

    print(f"{'model':<22} {'λ_opt':>6} {'L':>3} {'S̄':>7} {'CE':>8} {'|∂CE/∂g|':>10}  candidates")
    print("-" * 95)

    for name, fn in probes:
        stats = fn()
        emp = EMPIRICAL[name]
        L, S = emp["L"], emp["mean_S"]
        lam = emp["lambda_opt"]
        ce = stats["ce_mean"]
        g = stats["grad_mean"]

        # H3: λ ≈ k · g · S  (per-layer sparsity grad = λ/(L·S) ≈ g  →  λ ≈ L·S·g)
        h3 = L * S * g
        # H3 variant with L outside only
        h3b = S * g
        # H2: λ ≈ k · CE / S^α
        h2 = ce / S
        h2_sqrt = ce / (S ** 0.5)

        rows.append((name, lam, L, S, ce, g, h3, h3b, h2))

        print(f"{name:<22} {lam:>6.3f} {L:>3} {S:>7.0f} {ce:>8.4f} {g:>10.6f}  "
              f"L·S·g={h3:.4f}  CE/S={h2:.6f}")

    # Fit k for H3: λ = k · L · S · g
    import math
    ks_h3 = [lam / (L * S * g) for (_, lam, L, S, _, g, _, _, _) in rows]
    k_h3 = sum(ks_h3) / len(ks_h3)
    log_err_h3 = [abs(math.log10(lam) - math.log10(k_h3 * L * S * g)) for (_, lam, L, S, _, g, _, _, _) in rows]

    # Fit k for H2: λ = k · CE / S
    ks_h2 = [lam / (ce / S) for (_, lam, L, S, ce, g, _, _, _) in rows]
    k_h2 = sum(ks_h2) / len(ks_h2)
    log_err_h2 = [abs(math.log10(lam) - math.log10(k_h2 * ce / S)) for (_, lam, L, S, ce, g, _, _, _) in rows]

    # Fit k for combined: λ = k · L · CE / S
    ks_combo = [lam / (L * ce / S) for (_, lam, L, S, ce, g, _, _, _) in rows]
    k_combo = sum(ks_combo) / len(ks_combo)

    # Fit power law in log space: log λ = a + b·log(S) + c·log(CE) + d·log(L)
    import numpy as np
    X = np.array([[1, math.log(L), math.log(S), math.log(ce)] for (_, lam, L, S, ce, g, _, _, _) in rows])
    y = np.array([math.log(lam) for (_, lam, _, _, _, _, _, _, _) in rows])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, bL, bS, bCE = coef

    print("\n" + "=" * 95)
    print(f"H3 fit  λ ≈ k · L · S̄ · ⟨|∂CE/∂g|⟩     k = {k_h3:.4f}   mean |log10 err| = {sum(log_err_h3)/len(log_err_h3):.3f}")
    print(f"H2 fit  λ ≈ k · CE / S̄                k = {k_h2:.4f}   mean |log10 err| = {sum(log_err_h2)/len(log_err_h2):.3f}")
    print(f"Combo   λ ≈ k · L · CE / S̄            k = {k_combo:.4f}")
    print(f"Log-reg log λ = {a:.3f} + {bL:.3f}·log L + {bS:.3f}·log S + {bCE:.3f}·log CE")

    print("\nPredictions vs empirical:")
    print(f"{'model':<22} {'λ_emp':>7} {'H3':>7} {'H2':>7} {'combo':>7} {'logreg':>7}")
    for (name, lam, L, S, ce, g, _, _, _) in rows:
        pred_h3 = k_h3 * L * S * g
        pred_h2 = k_h2 * ce / S
        pred_c = k_combo * L * ce / S
        pred_lr = math.exp(a + bL*math.log(L) + bS*math.log(S) + bCE*math.log(ce))
        print(f"{name:<22} {lam:>7.3f} {pred_h3:>7.3f} {pred_h2:>7.3f} {pred_c:>7.3f} {pred_lr:>7.3f}")


if __name__ == "__main__":
    main()
