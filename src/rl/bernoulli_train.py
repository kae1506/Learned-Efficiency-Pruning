"""
Episode runner for the per-neuron Bernoulli action. Reuses
src.rl_ac_train.actor_critic_update for the gradient step (that update is
action-agnostic — it only consumes log_probs / entropies / rewards / values).

The only new piece is how the action is sampled and scored:
  - sample: b_j ~ Bernoulli(p_j) independently per alive neuron
  - log-prob: Σ_j [b_j log p_j + (1-b_j) log(1-p_j)]   (exact, factorised)
  - entropy: SUM of per-neuron Bernoulli entropies (scales with N_alive, the
    analog of the categorical log(N_alive) "raw" entropy that worked best;
    normalised/mean entropy regressed the result in the categorical study)
  - greedy: deterministically prune the round(Σ p_j) highest-p neurons, so the
    greedy rollout removes the same EXPECTED count the stochastic policy would
    (avoids degenerate 1-neuron-per-step greedy episodes when all p_j < 0.5).
"""

import torch
import torch.nn.functional as F


def run_episode_bernoulli(env, policy, value_net, greedy: bool = False):
    obs = env.reset()
    log_probs, entropies, rewards, values = [], [], [], []

    done = False
    while not done:
        neuron_feats, global_feat, alive_map = obs
        logits = policy(neuron_feats, global_feat)           # [N_alive]
        p = torch.sigmoid(logits)

        if greedy:
            # deterministic: prune the expected number of highest-p neurons
            n_prune = max(1, int(round(p.sum().item())))
            n_prune = min(n_prune, p.numel())
            chosen = torch.topk(p, n_prune).indices
            b = torch.zeros_like(p)
            b[chosen] = 1.0
        else:
            b = torch.bernoulli(p)
            if b.sum() == 0:                                  # guarantee progress
                b = torch.zeros_like(p)
                b[torch.argmax(p)] = 1.0

        # exact factorised Bernoulli log-prob (summed over alive neurons)
        log_p = (b * torch.log(p.clamp_min(1e-12)) +
                 (1 - b) * torch.log((1 - p).clamp_min(1e-12))).sum()
        # SUM of per-neuron Bernoulli entropies (scales with N_alive)
        entropy = -(p * torch.log(p.clamp_min(1e-12)) +
                    (1 - p) * torch.log((1 - p).clamp_min(1e-12))).sum()
        value = value_net(neuron_feats, global_feat)

        idx = b.nonzero(as_tuple=True)[0].tolist()
        obs, reward, done = env.step(idx)

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
