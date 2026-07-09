"""
PPO-trained pruning policy. Sister script to scripts/train_rl_pruner.py
(REINFORCE) — same env, same policy architecture, same reward, but uses
actor-critic + clipped PPO updates to suppress the variance that caused
REINFORCE's late-training policy collapse.

Run from project root:
    venv/bin/python scripts/train_rl_pruner_ppo.py
    venv/bin/python scripts/train_rl_pruner_ppo.py --max_prune 0.70 --tag 70
"""

import os
import sys
import argparse
import datetime
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(".")
from src.model import MLP
from src.dataset import get_mnist_loaders
from src.pruners.rl_policy import PolicyNet
from src.pruners.rl_value import ValueNet
from src.rl.env import PruneEnv
from src.rl.ppo_train import collect_episode, compute_gae, ppo_update


# ── config ────────────────────────────────────────────────────────────────────
CONFIG_PATH   = "configs/config.yaml"
CKPT_PATH     = "experiments/checkpoints/mnist_model.pt"
OUT_DIR       = "experiments/latest"

PRUNE_CHUNK        = 16
N_EPISODES         = 300
LR                 = 3e-4         # PPO standard; lower than REINFORCE 1e-3
CLIP_EPS           = 0.2
N_PPO_EPOCHS       = 4
VALUE_COEF         = 0.5
ENTROPY_COEF       = 0.01
GAMMA              = 1.0
LAM                = 0.95
MAX_GRAD_NORM      = 1.0
CALIB_BATCH        = 256
EVAL_BATCH         = 256
RECALIB_EVERY      = 5
GREEDY_EVAL_EVERY  = 25


def load_model(path: str, device) -> MLP:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = MLP(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def build_fixed_batches(loader, n_total: int, device):
    xs, ys = [], []
    it = iter(loader)
    grabbed = 0
    while grabbed < n_total:
        x, y = next(it)
        xs.append(x); ys.append(y)
        grabbed += x.size(0)
    return torch.cat(xs)[:n_total].to(device), torch.cat(ys)[:n_total].to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_prune", type=float, default=0.80)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    max_prune_fraction = args.max_prune
    tag = args.tag if args.tag is not None else f"{int(max_prune_fraction*100)}"
    run_dir      = f"{OUT_DIR}/rl/ppo/{tag}"
    plot_path    = f"{run_dir}/plot.png"
    summary_path = f"{run_dir}/summary.txt"

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  PPO  |  max_prune={max_prune_fraction:.2f}  |  tag={tag}")

    model = load_model(CKPT_PATH, device)
    train_loader, test_loader = get_mnist_loaders(**cfg["data"])

    calib_x, _      = build_fixed_batches(train_loader, CALIB_BATCH, device)
    eval_x, eval_y  = build_fixed_batches(test_loader,  EVAL_BATCH,  device)

    env = PruneEnv(
        model, calib_x, eval_x, eval_y, device,
        max_prune_fraction=max_prune_fraction,
        prune_chunk=PRUNE_CHUNK,
        recalibrate_every=RECALIB_EVERY,
    )
    print(f"Total hidden neurons: {env.total_neurons}  "
          f"(target: prune {int(env.total_neurons*max_prune_fraction)})")
    print(f"Orig acc on eval batch: {env.orig_acc*100:.2f}%")

    policy    = PolicyNet(env.feat_dim, env.global_dim, hidden=64).to(device)
    value_net = ValueNet (env.feat_dim, env.global_dim, hidden=64).to(device)
    opt = torch.optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=LR,
    )

    history = {
        "ep": [], "return": [], "final_acc": [], "acc_drop": [],
        "frac_pruned": [], "loss": [], "policy_loss": [], "value_loss": [], "entropy": [],
        "greedy_ep": [], "greedy_acc": [], "greedy_frac": [],
    }

    for ep in range(1, N_EPISODES + 1):
        transitions, info = collect_episode(env, policy, value_net, PRUNE_CHUNK)
        advs, returns = compute_gae(transitions, gamma=GAMMA, lam=LAM)
        stats = ppo_update(
            opt, policy, value_net, transitions, advs, returns,
            clip_eps=CLIP_EPS,
            n_epochs=N_PPO_EPOCHS,
            value_coef=VALUE_COEF,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
        )

        acc_drop = info["orig_acc"] - info["final_acc"]
        history["ep"].append(ep)
        history["return"].append(info["return"])
        history["final_acc"].append(info["final_acc"])
        history["acc_drop"].append(acc_drop)
        history["frac_pruned"].append(info["frac_pruned"])
        history["loss"].append(stats["loss"])
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])

        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:>4} | return={info['return']:+.4f} | "
                  f"final_acc={info['final_acc']*100:.2f}% | "
                  f"drop={acc_drop*100:.2f}% | "
                  f"frac_pruned={info['frac_pruned']*100:.1f}% | "
                  f"loss={stats['loss']:+.4f} | "
                  f"v_loss={stats['value_loss']:.4f} | "
                  f"ent={stats['entropy']:.3f}")

        if ep % GREEDY_EVAL_EVERY == 0:
            with torch.no_grad():
                _, g_info = collect_episode(env, policy, value_net, PRUNE_CHUNK, greedy=True)
            history["greedy_ep"].append(ep)
            history["greedy_acc"].append(g_info["final_acc"])
            history["greedy_frac"].append(g_info["frac_pruned"])
            print(f"    [greedy] final_acc={g_info['final_acc']*100:.2f}% | "
                  f"frac_pruned={g_info['frac_pruned']*100:.1f}%")

    # ── plot ──────────────────────────────────────────────────────────────────
    os.makedirs(run_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"RL Pruner — PPO  ({int(max_prune_fraction*100)}% target)",
                 fontsize=13, fontweight="bold")

    axes[0, 0].plot(history["ep"], history["return"], color="#2980b9", alpha=0.7)
    axes[0, 0].set_xlabel("Episode"); axes[0, 0].set_ylabel("Return")
    axes[0, 0].set_title("Episode return")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(history["ep"], [a*100 for a in history["final_acc"]],
                    color="#27ae60", alpha=0.6, label="sampled")
    if history["greedy_ep"]:
        axes[0, 1].plot(history["greedy_ep"], [a*100 for a in history["greedy_acc"]],
                        "o-", color="#1e8449", label="greedy")
    axes[0, 1].axhline(env.orig_acc * 100, color="k", ls="--", alpha=0.5, label="orig acc")
    axes[0, 1].set_xlabel("Episode"); axes[0, 1].set_ylabel("Final accuracy (%)")
    axes[0, 1].set_title(f"Final accuracy at ~{int(max_prune_fraction*100)}% pruned")
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[0, 2].plot(history["ep"], [d*100 for d in history["acc_drop"]],
                    color="#c0392b", alpha=0.7)
    axes[0, 2].set_xlabel("Episode"); axes[0, 2].set_ylabel("Accuracy drop (%)")
    axes[0, 2].set_title("Accuracy drop")
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].plot(history["ep"], history["policy_loss"], color="#8e44ad", alpha=0.7)
    axes[1, 0].set_xlabel("Episode"); axes[1, 0].set_ylabel("Policy loss")
    axes[1, 0].set_title("PPO policy (clipped) loss")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(history["ep"], history["value_loss"], color="#d35400", alpha=0.7)
    axes[1, 1].set_xlabel("Episode"); axes[1, 1].set_ylabel("Value loss")
    axes[1, 1].set_title("Critic MSE")
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].plot(history["ep"], history["entropy"], color="#16a085", alpha=0.7)
    axes[1, 2].set_xlabel("Episode"); axes[1, 2].set_ylabel("Entropy")
    axes[1, 2].set_title("Policy entropy")
    axes[1, 2].grid(alpha=0.3)

    fig.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {plot_path}")

    # ── summary ───────────────────────────────────────────────────────────────
    arch = (f"{cfg['model']['input_dim']} -> "
            + " -> ".join(str(d) for d in cfg['model']['hidden_dims'])
            + f" -> {cfg['model']['output_dim']}")

    last = -1
    best_idx = max(range(len(history["final_acc"])),
                   key=lambda i: history["final_acc"][i])

    lines = [
        "=" * 60,
        "RL PRUNER SUMMARY (PPO)",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "BASE MODEL",
        f"  Architecture        : MLP {arch}",
        f"  Checkpoint          : {CKPT_PATH}",
        f"  Total hidden        : {env.total_neurons} neurons",
        f"  Orig acc (eval batch): {env.orig_acc*100:.2f}%",
        "",
        "ENVIRONMENT",
        f"  Max prune fraction  : {max_prune_fraction*100:.0f}%",
        f"  Prune chunk         : {PRUNE_CHUNK}",
        f"  Recalibrate every   : {RECALIB_EVERY} steps",
        f"  Reward              : r_t = acc_t - acc_{{t-1}}",
        "",
        "ALGORITHM",
        f"  Method              : Actor-critic PPO (clipped)",
        f"  Clip epsilon        : {CLIP_EPS}",
        f"  PPO epochs / rollout: {N_PPO_EPOCHS}",
        f"  GAE lambda          : {LAM}",
        f"  Gamma               : {GAMMA}",
        f"  Value coef          : {VALUE_COEF}",
        f"  Entropy coef        : {ENTROPY_COEF}",
        f"  Max grad norm       : {MAX_GRAD_NORM}",
        "",
        "TRAINING",
        f"  Episodes            : {N_EPISODES}",
        f"  LR                  : {LR}",
        "",
        "RESULTS",
        f"  Final episode       : return={history['return'][last]:+.4f}  "
            f"acc={history['final_acc'][last]*100:.2f}%  "
            f"drop={history['acc_drop'][last]*100:.2f}%  "
            f"frac_pruned={history['frac_pruned'][last]*100:.1f}%",
        f"  Best-acc episode    : ep={history['ep'][best_idx]}  "
            f"acc={history['final_acc'][best_idx]*100:.2f}%  "
            f"drop={history['acc_drop'][best_idx]*100:.2f}%",
    ]
    if history["greedy_ep"]:
        lines += [
            "",
            "GREEDY EVAL",
            f"  Last greedy acc     : {history['greedy_acc'][-1]*100:.2f}% "
                f"(ep {history['greedy_ep'][-1]})",
            f"  Best greedy acc     : {max(history['greedy_acc'])*100:.2f}%",
        ]
    lines += ["", "=" * 60]

    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
