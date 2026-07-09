"""
Train the REINFORCE pruner at max_prune = 0.80, then run interpretability
analysis on the best-greedy mask seen during training.

Outputs (under experiments/latest/rl/reinforce_interp/80/):
  - training_curves.png   4-panel REINFORCE training plot
  - interp.png            interpretability panel (per-layer % pruned + histograms)
  - summary.txt           combined text report
  - run.log               full stdout

Run from project root:
    venv/bin/python scripts/train_rl_interp.py
"""

import os
import sys
import datetime
import yaml
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.rl_policy import PolicyNet
from src.rl.env import PruneEnv
from src.rl.train import run_episode, reinforce_update
from src.interpretability import analyze_pruner, evaluate_with_gates


# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH        = "configs/config.yaml"
CKPT_PATH          = "experiments/checkpoints/mnist_model.pt"
OUT_DIR            = "experiments/latest/rl/reinforce_interp/80"

MAX_PRUNE_FRACTION = 0.80
PRUNE_CHUNK        = 16
N_EPISODES         = 300
LR                 = 1e-3
ENTROPY_COEF       = 0.01
BASELINE_DECAY     = 0.95
GAMMA              = 1.0
CALIB_BATCH        = 256
EVAL_BATCH         = 256
RECALIB_EVERY      = 5
GREEDY_EVAL_EVERY  = 10   # frequent enough to catch the best greedy mask


def load_model(device) -> tuple[MLP, dict]:
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["config"]


def build_fixed_batches(loader, n_total: int, device):
    xs, ys = [], []
    it = iter(loader)
    grabbed = 0
    while grabbed < n_total:
        x, y = next(it)
        xs.append(x); ys.append(y)
        grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def _smooth(values, window=15):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def plot_training(history: dict, env, path: str):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"REINFORCE pruner — {int(MAX_PRUNE_FRACTION*100)}% prune target",
                 fontsize=13, fontweight="bold")

    axes[0,0].plot(history["ep"], history["return"], color="#2980b9", alpha=0.4)
    axes[0,0].plot(history["ep"], _smooth(history["return"]), color="#2980b9", lw=2)
    axes[0,0].set_xlabel("Episode"); axes[0,0].set_ylabel("Episode return")
    axes[0,0].set_title(r"Return  $\approx$  final acc $-$ orig acc")
    axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(history["ep"], [a*100 for a in history["final_acc"]],
                   color="#27ae60", alpha=0.4, label="sampled")
    axes[0,1].plot(history["ep"], _smooth([a*100 for a in history["final_acc"]]),
                   color="#27ae60", lw=2)
    if history["greedy_ep"]:
        axes[0,1].plot(history["greedy_ep"], [a*100 for a in history["greedy_acc"]],
                       "o-", color="#1e8449", label="greedy")
    axes[0,1].axhline(env.orig_acc * 100, color="k", ls="--", alpha=0.5,
                      label=f"orig {env.orig_acc*100:.2f}%")
    axes[0,1].set_xlabel("Episode"); axes[0,1].set_ylabel("Final accuracy (%)")
    axes[0,1].set_title(f"Final accuracy at ~{int(MAX_PRUNE_FRACTION*100)}% pruned")
    axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

    axes[1,0].plot(history["ep"], [d*100 for d in history["acc_drop"]],
                   color="#c0392b", alpha=0.4)
    axes[1,0].plot(history["ep"], _smooth([d*100 for d in history["acc_drop"]]),
                   color="#c0392b", lw=2)
    axes[1,0].set_xlabel("Episode"); axes[1,0].set_ylabel("Accuracy drop (%)")
    axes[1,0].set_title("Accuracy drop")
    axes[1,0].grid(alpha=0.3)

    axes[1,1].plot(history["ep"], history["loss"], color="#8e44ad", alpha=0.5)
    axes[1,1].plot(history["ep"], _smooth(history["loss"]), color="#8e44ad", lw=2)
    axes[1,1].set_xlabel("Episode"); axes[1,1].set_ylabel("Loss")
    axes[1,1].set_title("REINFORCE loss")
    axes[1,1].grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


def env_masks_to_gates(env) -> list[torch.Tensor]:
    """Convert env.masks (bool tensors, True = alive) to gates expected by
    analyze_pruner (any tensor where 1 = kept, 0 = pruned). The mask is already
    the right semantic — just clone to detach from the env."""
    return [m.clone().detach().cpu() for m in env.masks]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, mcfg = load_model(device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    calib_x, _     = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y = build_fixed_batches(test_loader,  EVAL_BATCH,  device)

    env = PruneEnv(
        model, calib_x, eval_x, eval_y, device,
        max_prune_fraction=MAX_PRUNE_FRACTION,
        prune_chunk=PRUNE_CHUNK,
        recalibrate_every=RECALIB_EVERY,
    )
    print(f"Total hidden neurons: {env.total_neurons}  "
          f"(target: prune {int(env.total_neurons*MAX_PRUNE_FRACTION)})")
    print(f"Orig acc on eval batch: {env.orig_acc*100:.2f}%\n")

    policy = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    opt    = torch.optim.Adam(policy.parameters(), lr=LR)

    baseline = 0.0
    history = {
        "ep": [], "return": [], "final_acc": [], "acc_drop": [],
        "frac_pruned": [], "loss": [],
        "greedy_ep": [], "greedy_acc": [], "greedy_frac": [],
    }

    best_greedy_acc   = -1.0
    best_greedy_ep    = 0
    best_greedy_gates = None

    for ep in range(1, N_EPISODES + 1):
        log_probs, entropies, rewards, info = run_episode(env, policy, PRUNE_CHUNK)
        ep_return = sum(rewards)
        loss = reinforce_update(opt, policy, log_probs, entropies, rewards,
                                baseline=baseline,
                                entropy_coef=ENTROPY_COEF,
                                gamma=GAMMA)
        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * ep_return

        acc_drop = info["orig_acc"] - info["final_acc"]
        history["ep"].append(ep)
        history["return"].append(ep_return)
        history["final_acc"].append(info["final_acc"])
        history["acc_drop"].append(acc_drop)
        history["frac_pruned"].append(info["frac_pruned"])
        history["loss"].append(loss)

        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:>4} | return={ep_return:+.4f} | "
                  f"final_acc={info['final_acc']*100:.2f}% | "
                  f"drop={acc_drop*100:.2f}% | "
                  f"frac_pruned={info['frac_pruned']*100:.1f}% | "
                  f"loss={loss:+.4f}")

        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, _, _, g_info = run_episode(env, policy, PRUNE_CHUNK, greedy=True)
            history["greedy_ep"].append(ep)
            history["greedy_acc"].append(g_info["final_acc"])
            history["greedy_frac"].append(g_info["frac_pruned"])
            if g_info["final_acc"] > best_greedy_acc:
                best_greedy_acc   = g_info["final_acc"]
                best_greedy_ep    = ep
                best_greedy_gates = env_masks_to_gates(env)
                print(f"    [greedy ★] ep={ep}  acc={g_info['final_acc']*100:.2f}%  "
                      f"frac_pruned={g_info['frac_pruned']*100:.1f}%  (new best)")
            else:
                print(f"    [greedy]   ep={ep}  acc={g_info['final_acc']*100:.2f}%")

    if best_greedy_gates is None:
        with torch.no_grad():
            _, _, _, g_info = run_episode(env, policy, PRUNE_CHUNK, greedy=True)
        best_greedy_acc   = g_info["final_acc"]
        best_greedy_ep    = N_EPISODES
        best_greedy_gates = env_masks_to_gates(env)

    plot_training(history, env, f"{OUT_DIR}/training_curves.png")

    # ── full-test-set evaluation of the best-greedy mask ──────────────────────
    test_acc = evaluate_with_gates(model, best_greedy_gates, test_loader, device)
    baseline_acc = evaluate_with_gates(
        model,
        [torch.ones_like(g) for g in best_greedy_gates],
        test_loader, device,
    )
    test_drop = baseline_acc - test_acc
    print(f"\nFull test set (best-greedy mask, ep {best_greedy_ep}):  "
          f"baseline={baseline_acc*100:.2f}%  pruned={test_acc*100:.2f}%  "
          f"drop={test_drop*100:.2f}pp\n")

    # ── interpretability ──────────────────────────────────────────────────────
    print(f"Running interpretability on best-greedy mask "
          f"(ep {best_greedy_ep}, greedy eval-batch acc {best_greedy_acc*100:.2f}%)")
    result = analyze_pruner(
        model=model,
        gates=best_greedy_gates,
        calib_loader=train_loader,
        device=device,
        n_calib_batches=5,
        save_plot=f"{OUT_DIR}/interp.png",
        verbose=True,
    )
    print(f"Saved interpretability plot to {OUT_DIR}/interp.png")

    # ── summary text file ─────────────────────────────────────────────────────
    summary_path = f"{OUT_DIR}/summary.txt"
    arch = (f"{mcfg['input_dim']} -> "
            + " -> ".join(str(d) for d in mcfg['hidden_dims'])
            + f" -> {mcfg['output_dim']}")
    lines = [
        "=" * 64,
        "REINFORCE PRUNER + INTERPRETABILITY",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 64,
        "",
        f"Base model         : MLP {arch}",
        f"Checkpoint         : {CKPT_PATH}",
        f"Max prune fraction : {MAX_PRUNE_FRACTION*100:.0f}%",
        f"Episodes           : {N_EPISODES}",
        f"Prune chunk        : {PRUNE_CHUNK}",
        f"Orig eval-batch acc: {env.orig_acc*100:.2f}%",
        "",
        f"Best greedy (256-sample eval batch): ep {best_greedy_ep}  "
        f"acc {best_greedy_acc*100:.2f}%  "
        f"(drop {(env.orig_acc - best_greedy_acc)*100:.2f}%)",
        "",
        f"Full test set     : baseline={baseline_acc*100:.2f}%  "
            f"pruned={test_acc*100:.2f}%  drop={test_drop*100:.2f}pp",
        "",
        f"OVERALL  pruned = {result['frac_pruned_total']*100:5.2f}%",
        "",
        "Per-layer breakdown:",
    ]
    for i, p in enumerate(result["per_layer"]):
        lines.append(
            f"  Layer {i+1}: {p['n_pruned']:>5}/{p['n_total']} pruned "
            f"({p['frac_pruned']*100:5.2f}%)"
        )
    lines += ["", "Mean post-ReLU activation (under ORIGINAL model):"]
    for i in range(len(result["per_layer"])):
        lines.append(
            f"  Layer {i+1}: alive {result['mean_act_alive_per_layer'][i]:.4f}  "
            f"dead {result['mean_act_dead_per_layer'][i]:.4f}"
        )
    lines += [
        "",
        f"OVERALL  alive mean activation = {result['mean_act_alive_overall']:.4f}",
        f"OVERALL  dead  mean activation = {result['mean_act_dead_overall']:.4f}",
    ]
    if result["mean_act_alive_overall"]:
        ratio = result["mean_act_dead_overall"] / result["mean_act_alive_overall"]
        lines.append(f"OVERALL  ratio (dead/alive)    = {ratio:.3f}")
    lines += ["", "=" * 64]
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
