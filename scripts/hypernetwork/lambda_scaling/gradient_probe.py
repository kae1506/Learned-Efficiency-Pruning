"""
Gradient probe — measure |∂CE/∂g_i| at gates=1 on each frozen base model.

Question: can we predict λ_opt from one cheap forward+backward pass per model,
without sweeping λ?

Hypothesis (from sw_formula.md): at the pruner's optimum, the per-gate sparsity
gradient λ_opt/(N_layers·S_ℓ) balances the per-gate CE gradient |∂CE/∂g_i| at
the boundary. If the CE gradient distribution at gates=1 is a good proxy, then
λ_opt should be predictable from some statistic of {|∂CE/∂g_i|}.

Probe protocol per base model:
  • Freeze model (eval mode, no grad on weights).
  • For 2 training batches × 256 samples:
      - Create gate vectors per layer, all-ones, requires_grad=True
      - Run that model's masked_forward (so gates are differentiable scalars
        applied to W rows + biases — exactly the path the pruner trains through).
      - CE = cross_entropy(logits, y) ; CE.backward() ; collect g.grad.abs()
  • Average gradients across the 2 batches per layer.
  • Compute candidate predictors (P1-P5) and fit log-LS constant c to existing
    observed λ_opt values across 5 nets.

Skips MNIST Narrow [205×2] per user request — only 5 nets.
Probe data = TRAINING data (matches what the pruner sees).

Run from project root:
  venv/bin/python scripts/hypernetwork/gradient_probe.py
"""

import os
import sys
import math
import json

sys.path.append(".")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-architecture base classes / forward paths
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.prune_train import get_hidden_weights, masked_forward as masked_fwd_mlp
from scripts.base.train_cifar import CIFARNet, CIFARNetBig, CIFAR_MEAN, CIFAR_STD
from scripts.hypernetwork.train.train_pruner_cifar_lenet import (
    masked_forward as masked_fwd_lenet,
    get_fc_weights  as get_fc_weights_lenet,
)
from scripts.hypernetwork.train.train_pruner_cifar import (
    masked_forward as masked_fwd_big,
    get_fc_weights  as get_fc_weights_big,
)


OUT_DIR    = "experiments/latest/hypernetwork/gradient_probe"
N_BATCHES  = 2          # per user request — average over 2 batches per probe
BATCH_SIZE = 256


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar_train_loader(batch_size: int = BATCH_SIZE):
    tf = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train = torchvision.datasets.CIFAR10(root="./data", train=True,
                                         download=True, transform=tf)
    return torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0)


# ─────────────────────────────────────────────────────────────────────────────
# Per-arch setup functions — return (frozen model, layer-weight getter,
# masked_forward fn, train_loader)
# ─────────────────────────────────────────────────────────────────────────────

def setup_mnist(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    train_loader, _ = get_mnist_loaders(data_dir="./data", batch_size=BATCH_SIZE)
    return model, get_hidden_weights, masked_fwd_mlp, train_loader


def setup_cifar_lenet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CIFARNet(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, get_fc_weights_lenet, masked_fwd_lenet, get_cifar_train_loader()


def setup_cifar_big(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = CIFARNetBig(output_dim=ckpt["config"]["output_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, get_fc_weights_big, masked_fwd_big, get_cifar_train_loader()


# (label, N_L, observed_λ_opt, setup_fn, ckpt_path)
# Narrow [205×2] excluded per user instruction.
MODELS = [
    ("MNIST Wide [2048]",     1, 0.03, setup_mnist,       "experiments/checkpoints/mnist_wide2048.pt"),
    ("MNIST Medium [1024x2]", 2, 0.06, setup_mnist,       "experiments/checkpoints/mnist_model.pt"),
    ("MNIST Deep [512x4]",    4, 0.25, setup_mnist,       "experiments/checkpoints/mnist_deep4x512.pt"),
    ("LeNet (CIFAR)",         2, 0.04, setup_cifar_lenet, "experiments/checkpoints/cifar_cnn.pt"),
    ("CIFAR_big",             3, 0.03, setup_cifar_big,   "experiments/checkpoints/cifar_big.pt"),
]


# ─────────────────────────────────────────────────────────────────────────────
# The probe: avg |∂CE/∂g_i| over N_BATCHES TRAINING batches with gates=1
# ─────────────────────────────────────────────────────────────────────────────

def probe_one(model, get_layer_weights, masked_fwd, loader, device):
    """Returns (grad_per_layer, layer_shapes). grad_per_layer[ℓ] = mean |∂CE/∂g| over batches."""
    layer_weights = get_layer_weights(model)
    layer_shapes  = [(w.shape[0], w.shape[1]) for w in layer_weights]

    grad_acc = [torch.zeros(out_size, device=device) for out_size, _ in layer_shapes]
    n_done = 0
    data_iter = iter(loader)
    while n_done < N_BATCHES:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)

        # Fresh gate tensors with requires_grad each batch — backward populates .grad
        gates  = [torch.ones(out_size, device=device, requires_grad=True)
                  for out_size, _ in layer_shapes]
        logits = masked_fwd(model, gates, x)
        # Note: at gates=1 this equals the unpruned forward, but the gradient
        # path through `g · weight` is well-defined.
        ce = F.cross_entropy(logits, y)
        ce.backward()

        for i, g in enumerate(gates):
            grad_acc[i] += g.grad.detach().abs()
        n_done += 1

    return [acc / n_done for acc in grad_acc], layer_shapes


# ─────────────────────────────────────────────────────────────────────────────
# Candidate predictor formulas — P1..P5 from the design doc
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(grad_per_layer, layer_shapes, N_L):
    per_layer_mean   = [g.mean().item()                  for g in grad_per_layer]
    per_layer_std    = [g.std().item()                   for g in grad_per_layer]
    per_layer_med    = [g.median().item()                for g in grad_per_layer]
    per_layer_p25    = [torch.quantile(g, 0.25).item()   for g in grad_per_layer]
    per_layer_p75    = [torch.quantile(g, 0.75).item()   for g in grad_per_layer]

    all_grads = torch.cat(grad_per_layer)
    mean_overall   = all_grads.mean().item()
    std_overall    = all_grads.std().item()
    median_overall = all_grads.median().item()

    S_avg     = sum(s[0] for s in layer_shapes) / len(layer_shapes)
    per_layer_NSm  = [N_L * s[0] * m for s, m in zip(layer_shapes, per_layer_mean)]
    per_layer_NSmd = [N_L * s[0] * m for s, m in zip(layer_shapes, per_layer_med)]

    return {
        # Raw distributional stats (for diagnostics)
        "per_layer_mean":   per_layer_mean,
        "per_layer_std":    per_layer_std,
        "per_layer_median": per_layer_med,
        "per_layer_p25":    per_layer_p25,
        "per_layer_p75":    per_layer_p75,
        "mean_overall":     mean_overall,
        "std_overall":      std_overall,
        "median_overall":   median_overall,
        # Candidate predictors
        "P1_value": N_L * S_avg * mean_overall,                   # mean over all gates
        "P2_value": max(per_layer_NSm),                           # max layer · N · S · ⟨|g|⟩
        "P3_value": sorted(per_layer_NSm)[len(per_layer_NSm)//2], # median layer
        "P4_value": max(per_layer_NSmd),                          # bottleneck-layer median
        "P5_value": N_L * sum(s[0] * m for s, m in zip(layer_shapes, per_layer_mean)),  # sum-weighted
    }


def fit_log_lsq(predictors, observed):
    """c that minimises Σ (log(obs_i) − log(c·pred_i))² → log_c = mean(log obs − log pred)."""
    log_obs  = np.log(np.array(observed, dtype=float))
    log_pred = np.log(np.array(predictors, dtype=float))
    return float(np.exp((log_obs - log_pred).mean()))


# ─────────────────────────────────────────────────────────────────────────────
# Main: probe each model, fit each formula, report + plot
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    print(f"Device: {device}  N_BATCHES={N_BATCHES}  BATCH_SIZE={BATCH_SIZE}")

    results = []
    for label, N_L, lambda_opt, setup_fn, ckpt_path in MODELS:
        print(f"\n── {label} ──  (N_L={N_L}, observed λ_opt={lambda_opt})", flush=True)
        model, get_lw, masked_fwd, loader = setup_fn(ckpt_path, device)
        grad_per_layer, layer_shapes = probe_one(model, get_lw, masked_fwd, loader, device)
        features = compute_features(grad_per_layer, layer_shapes, N_L)
        results.append({"label": label, "N_L": N_L, "lambda_opt": lambda_opt,
                        "layer_shapes": layer_shapes, "features": features})
        plm = features["per_layer_mean"]
        print(f"  layer_shapes      = {layer_shapes}")
        print(f"  per-layer ⟨|g|⟩   = {['%.3e' % v for v in plm]}")
        print(f"  overall ⟨|g|⟩     = {features['mean_overall']:.3e}")
        print(f"  P1 (N·S_avg·⟨|g|⟩)            = {features['P1_value']:.3e}")
        print(f"  P2 (max ℓ of N·S_ℓ·⟨|g|⟩_ℓ)   = {features['P2_value']:.3e}")
        print(f"  P3 (median ℓ of N·S_ℓ·⟨|g|⟩_ℓ)= {features['P3_value']:.3e}")
        print(f"  P4 (max ℓ of N·S_ℓ·median_ℓ)  = {features['P4_value']:.3e}")
        print(f"  P5 (N·Σ_ℓ S_ℓ·⟨|g|⟩_ℓ)        = {features['P5_value']:.3e}")

    # ── Fit each formula ──────────────────────────────────────────────────────
    formulas = ["P1_value", "P2_value", "P3_value", "P4_value", "P5_value"]
    formula_names = {
        "P1_value": "P1: c · N · S_avg · ⟨|g|⟩",
        "P2_value": "P2: c · max_ℓ (N · S_ℓ · ⟨|g|⟩_ℓ)",
        "P3_value": "P3: c · median_ℓ (N · S_ℓ · ⟨|g|⟩_ℓ)",
        "P4_value": "P4: c · max_ℓ (N · S_ℓ · median(|g|_ℓ))",
        "P5_value": "P5: c · N · Σ_ℓ S_ℓ · ⟨|g|⟩_ℓ",
    }

    obs    = [r["lambda_opt"] for r in results]
    labels = [r["label"]      for r in results]
    fit_summary = []
    for f in formulas:
        preds = [r["features"][f] for r in results]
        c     = fit_log_lsq(preds, obs)
        pred_lambdas = [c * p for p in preds]
        residuals    = [math.log(o / pl) for o, pl in zip(obs, pred_lambdas)]
        max_resid    = max(abs(r) for r in residuals)
        rms_resid    = math.sqrt(sum(r*r for r in residuals) / len(residuals))
        log_obs   = np.log(obs)
        ss_res    = sum((np.log(obs[i]) - np.log(pred_lambdas[i]))**2 for i in range(len(obs)))
        ss_tot    = sum((log_obs[i] - log_obs.mean())**2 for i in range(len(obs)))
        r2        = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
        fit_summary.append({"formula": f, "name": formula_names[f],
                            "c": c, "preds": pred_lambdas, "residuals": residuals,
                            "max_residual": max_resid, "rms_residual": rms_resid,
                            "r2_log": r2})

    # ── Print + write text summary ────────────────────────────────────────────
    lines = ["=" * 88,
             "GRADIENT PROBE — predicted λ_opt vs observed (5 nets; narrow excluded)",
             "=" * 88,
             f"Protocol: {N_BATCHES} TRAINING batches × {BATCH_SIZE} samples, gates=1, CPU",
             ""]
    lines.append(f"{'model':<25} {'observed λ_opt':>15}")
    for label, o in zip(labels, obs):
        lines.append(f"{label:<25} {o:>15.4f}")
    lines.append("")
    for fs in fit_summary:
        lines.append(f"--- {fs['name']} ---  c={fs['c']:.4e}  "
                     f"R²(log)={fs['r2_log']:.3f}  "
                     f"max ratio={math.exp(fs['max_residual']):.2f}×  "
                     f"RMS ratio={math.exp(fs['rms_residual']):.2f}×")
        for label, o, p in zip(labels, obs, fs["preds"]):
            lines.append(f"  {label:<25}  observed={o:.4f}  predicted={p:.4f}  "
                         f"ratio={o/p:.2f}×")
        lines.append("")

    text = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(text)
    print("\n" + text)

    # ── Plot: 5-panel observed-vs-predicted, log-log ──────────────────────────
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
    for ax, fs in zip(axes, fit_summary):
        ax.scatter(fs["preds"], obs, s=110, color="steelblue", zorder=3)
        for label, o, p in zip(labels, obs, fs["preds"]):
            tag = (label.replace("MNIST ", "").replace("CIFAR ", "")
                        .replace("(CIFAR)", "").strip())[:10]
            ax.annotate(tag, (p, o), xytext=(5, 5),
                        textcoords="offset points", fontsize=8)
        all_pts = list(fs["preds"]) + list(obs)
        lo, hi  = min(all_pts) * 0.5, max(all_pts) * 2
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("predicted λ"); ax.set_ylabel("observed λ_opt")
        ax.set_title(f"{fs['name']}\nR²(log)={fs['r2_log']:.2f}  "
                     f"max ratio={math.exp(fs['max_residual']):.1f}×",
                     fontsize=10)
        ax.grid(alpha=0.3, which="both")
    fig.suptitle("Gradient probe: predicted vs observed λ_opt across 5 base models",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR, "scaling_fits.png"),
                                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Raw probe data → JSON for later analysis ──────────────────────────────
    json_dump = []
    for r in results:
        json_dump.append({
            "label":        r["label"],
            "N_L":          r["N_L"],
            "lambda_opt":   r["lambda_opt"],
            "layer_shapes": r["layer_shapes"],
            "features":     {k: v for k, v in r["features"].items()
                             if not isinstance(v, list) or isinstance(v[0], (int, float))},
        })
    with open(os.path.join(OUT_DIR, "probe_results.json"), "w") as f:
        json.dump(json_dump, f, indent=2)
    print(f"\nResults → {OUT_DIR}/  (summary.txt, scaling_fits.png, probe_results.json)")


if __name__ == "__main__":
    main()
