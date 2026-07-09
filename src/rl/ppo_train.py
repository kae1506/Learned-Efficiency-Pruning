"""
PPO trainer for the prune-MDP. Drop-in replacement for src/rl_train.py
(REINFORCE) with the following variance-reducing additions:

  - Actor-critic: a value head produces per-state V(s); advantages are
    A_t = GAE(r_t, V(s_t), V(s_{t+1})) instead of (R_t - EMA_baseline).
  - PPO clipped objective: bounds the policy update by a probability
    ratio clip, preventing the destructive large updates that caused
    REINFORCE's ep-200 policy collapse.
  - Multiple optimisation epochs per rollout: each collected episode is
    re-used n_epochs times, improving sample efficiency.

The state is variable-length (alive-set shrinks per step), so rollout
transitions are processed one-at-a-time during the update rather than
batched into tensors. This is slower per step than a fixed-size PPO
setup but keeps the code simple and avoids padding logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── rollout ────────────────────────────────────────────────────────────────────

def collect_episode(env, policy, value_net, prune_chunk: int, greedy: bool = False):
    """
    Run one episode, recording per-step transitions for PPO.

    Each transition stores the (detached) observation, the action taken,
    the old-policy log-prob of that action, the value estimate, and the
    immediate reward. Episode terminates when env signals done.
    """
    obs = env.reset()
    transitions = []

    done = False
    while not done:
        neuron_feats, global_feat, alive_map = obs

        with torch.no_grad():
            logits = policy(neuron_feats, global_feat)
            probs  = F.softmax(logits, dim=-1)
            k = min(prune_chunk, len(alive_map))
            if greedy:
                action = torch.topk(probs, k).indices
            else:
                action = torch.multinomial(probs, k, replacement=False)
            log_prob_old = torch.log(probs[action].clamp_min(1e-12)).sum()
            value = value_net(neuron_feats, global_feat)

        next_obs, reward, done = env.step(action.tolist())

        transitions.append({
            "neuron_feats" : neuron_feats.detach(),
            "global_feat"  : global_feat.detach(),
            "action"       : action.detach(),
            "log_prob_old" : log_prob_old.detach(),
            "value"        : value.detach(),
            "reward"       : reward,
            "done"         : done,
        })
        obs = next_obs

    info = {
        "final_acc"   : env.prev_acc,
        "orig_acc"    : env.orig_acc,
        "frac_pruned" : env.fraction_pruned,
        "n_steps"     : len(transitions),
        "return"      : sum(t["reward"] for t in transitions),
    }
    return transitions, info


# ── GAE ────────────────────────────────────────────────────────────────────────

def compute_gae(transitions, gamma: float = 1.0, lam: float = 0.95):
    """
    Generalised Advantage Estimation.

    delta_t  = r_t + gamma * V(s_{t+1}) - V(s_t)
    A_t      = delta_t + (gamma * lam) * A_{t+1}
    return_t = A_t + V(s_t)

    Episode terminates at the last transition (no bootstrap value).
    """
    advantages = [0.0] * len(transitions)
    gae = 0.0
    for t in reversed(range(len(transitions))):
        v_t      = transitions[t]["value"].item()
        v_next   = 0.0 if t == len(transitions) - 1 else transitions[t + 1]["value"].item()
        delta    = transitions[t]["reward"] + gamma * v_next - v_t
        gae      = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [advantages[t] + transitions[t]["value"].item() for t in range(len(transitions))]
    return advantages, returns


# ── PPO update ─────────────────────────────────────────────────────────────────

def ppo_update(
    optimizer,
    policy,
    value_net,
    transitions: list[dict],
    advantages: list[float],
    returns:    list[float],
    clip_eps:     float = 0.2,
    n_epochs:     int   = 4,
    value_coef:   float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 1.0,
):
    """
    Multi-epoch PPO update over a single episode's transitions.

    Loss per transition:
        L_policy  = -min( r * A, clip(r, 1-eps, 1+eps) * A )
        L_value   = (V(s) - return)^2
        L_entropy = -H(pi(.|s))
        loss = L_policy + value_coef * L_value + entropy_coef * L_entropy

    Advantages are standardised within the rollout for stability.
    """
    device = transitions[0]["neuron_feats"].device
    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    ret_t = torch.tensor(returns,    dtype=torch.float32, device=device)

    n = len(transitions)
    total_loss   = 0.0
    total_policy = 0.0
    total_value  = 0.0
    total_ent    = 0.0
    n_updates    = 0

    for _ in range(n_epochs):
        order = torch.randperm(n)
        for i in order.tolist():
            tr = transitions[i]

            logits = policy(tr["neuron_feats"], tr["global_feat"])
            probs  = F.softmax(logits, dim=-1)
            log_prob_new = torch.log(probs[tr["action"]].clamp_min(1e-12)).sum()
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum()
            value   = value_net(tr["neuron_feats"], tr["global_feat"])

            ratio  = torch.exp(log_prob_new - tr["log_prob_old"])
            adv    = adv_t[i]
            ret    = ret_t[i]

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2)
            value_loss  = (value - ret).pow(2)
            ent_bonus   = -entropy_coef * entropy

            loss = policy_loss + value_coef * value_loss + ent_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(policy.parameters()) + list(value_net.parameters()),
                max_grad_norm,
            )
            optimizer.step()

            total_loss   += loss.item()
            total_policy += policy_loss.item()
            total_value  += value_loss.item()
            total_ent    += entropy.item()
            n_updates    += 1

    return {
        "loss"        : total_loss   / n_updates,
        "policy_loss" : total_policy / n_updates,
        "value_loss"  : total_value  / n_updates,
        "entropy"     : total_ent    / n_updates,
    }
