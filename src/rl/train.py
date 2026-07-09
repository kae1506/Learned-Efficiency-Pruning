import torch
import torch.nn.functional as F


def run_episode(env, policy, prune_chunk: int, greedy: bool = False):
    """
    Run one episode to termination. Returns (log_prob_sum, entropy_sum, rewards, info).
    log_prob and entropy are summed over macro-steps; rewards is the per-step list.
    """
    obs = env.reset()
    log_probs, entropies, rewards = [], [], []

    done = False
    while not done:
        neuron_feats, global_feat, alive_map = obs
        logits = policy(neuron_feats, global_feat)
        probs = F.softmax(logits, dim=-1)

        k = min(prune_chunk, len(alive_map))
        if greedy:
            sampled = torch.topk(probs, k).indices
            log_p   = torch.log(probs[sampled].clamp_min(1e-12)).sum()
        else:
            sampled = torch.multinomial(probs, k, replacement=False)
            log_p   = torch.log(probs[sampled].clamp_min(1e-12)).sum()

        entropy = -(probs * probs.clamp_min(1e-12).log()).sum()

        obs, reward, done = env.step(sampled.tolist())
        log_probs.append(log_p)
        entropies.append(entropy)
        rewards.append(reward)

    info = {
        "final_acc": env.prev_acc,
        "orig_acc":  env.orig_acc,
        "frac_pruned": env.fraction_pruned,
        "n_steps":   len(rewards),
    }
    return log_probs, entropies, rewards, info


def reinforce_update(
    optimizer,
    policy,
    log_probs: list[torch.Tensor],
    entropies: list[torch.Tensor],
    rewards: list[float],
    baseline: float,
    entropy_coef: float,
    gamma: float = 1.0,
):
    device = log_probs[0].device
    returns = []
    R = 0.0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.append(R)
    returns.reverse()
    returns_t = torch.tensor(returns, device=device, dtype=torch.float32)
    adv = returns_t - baseline

    lp = torch.stack(log_probs)
    ent = torch.stack(entropies)
    loss = -(lp * adv).mean() - entropy_coef * ent.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return loss.item()
