"""
Actor-critic trainer for the prune-MDP. Drop-in replacement for src/rl_train.py
(REINFORCE) with two specific variance-reduction changes:

  1. Learned V(s) baseline. Replace the EMA scalar baseline with a value head
     that predicts the expected return from each state. Advantages become
     A_t = R_t - V(s_t), which is per-state and zero-mean in expectation.

  2. Reward-to-go for the policy gradient. Step t's gradient is scaled by
     R_t = Σ_{k≥t} γ^(k-t) r_k — only the rewards that step t could have
     influenced, not the full episode return. (src/rl_train.py technically
     already uses this through its discounted-return loop, but with a
     single scalar baseline; here both pieces fit together.)

No PPO clipping, no multi-epoch reuse. Single on-policy update per episode.
The intent is to isolate the variance reduction from V(s) alone, separately
from PPO's other tricks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def run_episode_ac(env, policy, value_net, prune_chunk: int, greedy: bool = False,
                   normalize_entropy: bool = False):
    """
    Run one episode, recording (log_probs, entropies, rewards, values, info).
    Gradients flow through log_probs and values into policy / value_net.

    normalize_entropy: divide each step's entropy by log(N_alive) so it lies in
      [0, 1] regardless of action-space size. Without this the raw entropy is
      log(N_alive) at uniform (≈7.6 at 2048 alive, ≈6.0 at 410), so a fixed
      entropy_coef applies wildly different pressure at the start vs end of an
      episode and across model sizes. Normalising decouples the entropy bonus
      from the live-set size.
    """
    obs = env.reset()
    log_probs, entropies, rewards, values = [], [], [], []

    done = False
    while not done:
        neuron_feats, global_feat, alive_map = obs
        logits = policy(neuron_feats, global_feat)
        probs  = F.softmax(logits, dim=-1)

        k = min(prune_chunk, len(alive_map))
        if greedy:
            sampled = torch.topk(probs, k).indices
        else:
            sampled = torch.multinomial(probs, k, replacement=False)
        log_p   = torch.log(probs[sampled].clamp_min(1e-12)).sum()
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
        if normalize_entropy:
            n_act = probs.numel()
            if n_act > 1:
                entropy = entropy / math.log(n_act)
        value   = value_net(neuron_feats, global_feat)

        obs, reward, done = env.step(sampled.tolist())
        log_probs.append(log_p)
        entropies.append(entropy)
        rewards.append(reward)
        values.append(value)

    info = {
        "final_acc"  : env.prev_acc,
        "orig_acc"   : env.orig_acc,
        "frac_pruned": env.fraction_pruned,
        "n_steps"    : len(rewards),
        "return"     : sum(rewards),
    }
    return log_probs, entropies, rewards, values, info


def actor_critic_update(
    optimizer,
    policy,
    value_net,
    log_probs: list[torch.Tensor],
    entropies: list[torch.Tensor],
    rewards: list[float],
    values: list[torch.Tensor],
    entropy_coef: float = 0.01,
    value_coef:   float = 0.5,
    gamma:        float = 1.0,
    max_grad_norm: float = 1.0,
):
    """
    Single on-policy update. Loss has three terms:

        L_policy  = -mean( log_prob_t · A_t )          A_t = R_t - V(s_t).detach()
        L_value   = MSE( V(s_t), R_t )
        L_entropy = -entropy_coef · mean( H(π_t) )

        loss = L_policy + value_coef · L_value + L_entropy
    """
    device = log_probs[0].device

    # ── reward-to-go: R_t = Σ_{k>=t} γ^(k-t) r_k ─────────────────────────────
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    returns_t = torch.tensor(returns, device=device, dtype=torch.float32)

    values_t   = torch.stack(values)                       # critic predictions
    advantages = (returns_t - values_t).detach()           # detach for policy grad

    lp  = torch.stack(log_probs)
    ent = torch.stack(entropies)

    policy_loss  = -(lp * advantages).mean()
    value_loss   = F.mse_loss(values_t, returns_t)
    entropy_term = -entropy_coef * ent.mean()

    loss = policy_loss + value_coef * value_loss + entropy_term

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(policy.parameters()) + list(value_net.parameters()),
        max_grad_norm,
    )
    optimizer.step()

    return {
        "loss"        : loss.item(),
        "policy_loss" : policy_loss.item(),
        "value_loss"  : value_loss.item(),
        "entropy"     : ent.mean().item(),
    }
