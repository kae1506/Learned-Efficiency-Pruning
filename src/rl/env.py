import torch
import torch.nn as nn
import torch.nn.functional as F


class PruneEnv:
    """
    MDP for sequential neuron pruning on a frozen MLP.

    State: per-neuron features for every still-alive hidden neuron, plus a
    small global trajectory vector. Action: a set of `prune_chunk` neurons
    to shut off (sampled without replacement from the policy). Episode
    terminates when fraction_pruned >= max_prune_fraction.

    Reward (per macro-step): r_t = acc_t - acc_{t-1}.
    Telescopes to (final_acc - orig_acc), so total return measures how
    much accuracy survived the full 80% prune schedule.
    """

    PER_NEURON_FEATS = 5    # in_l1, in_l2, out_l1, out_l2, mean_act
    GLOBAL_FEATS     = 3    # ce_gap, frac_pruned_total, current_acc

    def __init__(
        self,
        model: nn.Module,
        calib_x: torch.Tensor,
        eval_x: torch.Tensor,
        eval_y: torch.Tensor,
        device,
        max_prune_fraction: float = 0.8,
        prune_chunk: int = 16,
        recalibrate_every: int = 5,
    ):
        self.model = model
        self.device = device
        self.max_prune_fraction = max_prune_fraction
        self.prune_chunk = prune_chunk
        self.recalibrate_every = recalibrate_every

        self.calib_x = calib_x.view(calib_x.size(0), -1).to(device)
        self.eval_x  = eval_x.view(eval_x.size(0), -1).to(device)
        self.eval_y  = eval_y.to(device)

        self.linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.hidden_linears = self.linears[:-1]
        self.layer_sizes = [L.out_features for L in self.hidden_linears]
        self.n_layers = len(self.hidden_linears)
        self.total_neurons = sum(self.layer_sizes)
        self.feat_dim   = self.PER_NEURON_FEATS + self.n_layers + 1   # + one-hot layer + frac_pruned_layer
        self.global_dim = self.GLOBAL_FEATS

        with torch.no_grad():
            self.in_l1 = [L.weight.abs().sum(dim=1)            for L in self.hidden_linears]
            self.in_l2 = [L.weight.pow(2).sum(dim=1).sqrt()    for L in self.hidden_linears]

            logits = model(self.eval_x)
            self.orig_ce  = F.cross_entropy(logits, self.eval_y).item()
            self.orig_acc = (logits.argmax(1) == self.eval_y).float().mean().item()

        self._alive_map = None  # cached (layer, neuron) per alive index

    # ── public API ───────────────────────────────────────────────────────────

    def reset(self):
        self.masks = [torch.ones(s, dtype=torch.bool, device=self.device)
                      for s in self.layer_sizes]
        self.step_count = 0
        self.prev_acc = self.orig_acc
        self.prev_ce  = self.orig_ce
        self._update_activations()
        return self._observe()

    def step(self, action_indices: list[int]):
        """
        action_indices: indices into the alive list returned by the most recent
        observation. Each one is converted to (layer, local_idx) and that neuron
        is masked off. Returns (obs, reward, done).
        """
        assert self._alive_map is not None, "must call reset() / observe before step()"
        for idx in action_indices:
            # _alive_map converts idx to layer, neuron. 
            layer_i, local_j = self._alive_map[idx]
            self.masks[layer_i][local_j] = False

        self.step_count += 1
        if self.step_count % self.recalibrate_every == 0:
            self._update_activations()

        new_acc, new_ce = self._eval_acc_ce()
        reward = new_acc - self.prev_acc
        self.prev_acc = new_acc
        self.prev_ce  = new_ce

        # self.masks has shape (neurons, layers)
        frac_pruned = 1.0 - sum(m.sum().item() for m in self.masks) / self.total_neurons
        done = frac_pruned >= self.max_prune_fraction

        return self._observe(), reward, done

    @property
    def fraction_pruned(self) -> float:
        return 1.0 - sum(m.sum().item() for m in self.masks) / self.total_neurons

    # ── internals ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _update_activations(self):
        h = self.calib_x
        self.mean_acts = []
        for i, L in enumerate(self.hidden_linears):
            g = self.masks[i].float()
            w = L.weight * g.unsqueeze(1)
            b = L.bias   * g
            h = F.relu(F.linear(h, w, b))
            self.mean_acts.append(h.mean(dim=0))

    @torch.no_grad()
    def _eval_acc_ce(self):
        h = self.eval_x
        for i, L in enumerate(self.hidden_linears):
            g = self.masks[i].float()
            w = L.weight * g.unsqueeze(1)
            b = L.bias   * g
            h = F.relu(F.linear(h, w, b))
        out_L = self.linears[-1]
        logits = F.linear(h, out_L.weight, out_L.bias)
        ce  = F.cross_entropy(logits, self.eval_y).item()
        acc = (logits.argmax(1) == self.eval_y).float().mean().item()
        return acc, ce

    @torch.no_grad()
    def _outgoing_norms(self):
        out_l1, out_l2 = [], []
        for i in range(self.n_layers):
            downstream = self.linears[i + 1].weight       # [out, in_i]
            if i + 1 < self.n_layers:
                ds_alive = self.masks[i + 1].float().unsqueeze(1)
                w_eff = downstream * ds_alive             # zero rows whose downstream neuron is dead
            else:
                w_eff = downstream
            out_l1.append(w_eff.abs().sum(dim=0))         # per-input-column
            out_l2.append(w_eff.pow(2).sum(dim=0).sqrt())
        return out_l1, out_l2

    def _observe(self):
        out_l1, out_l2 = self._outgoing_norms()

        feats = []
        alive_map: list[tuple[int, int]] = []

        for i in range(self.n_layers):
            alive = self.masks[i]
            if not alive.any():
                continue
            idx = alive.nonzero(as_tuple=True)[0]
            n_alive = idx.numel()

            per_neuron = torch.stack([
                self.in_l1[i][idx],
                self.in_l2[i][idx],
                out_l1[i][idx],
                out_l2[i][idx],
                self.mean_acts[i][idx],
            ], dim=1)

            one_hot = torch.zeros(n_alive, self.n_layers, device=self.device)
            one_hot[:, i] = 1.0
            frac_pruned_layer = 1.0 - alive.float().mean().item()
            frac_col = torch.full((n_alive, 1), frac_pruned_layer, device=self.device)

            feats.append(torch.cat([per_neuron, one_hot, frac_col], dim=1))
            alive_map.extend((i, j) for j in idx.tolist())

        neuron_feats = torch.cat(feats, dim=0)

        # Light feature normalisation: in/out norms vary by 1-2 orders of magnitude.
        # Per-feature standardise across the alive set (cheap; keeps logits stable).
        with torch.no_grad():
            mu = neuron_feats[:, :self.PER_NEURON_FEATS].mean(dim=0, keepdim=True)
            sd = neuron_feats[:, :self.PER_NEURON_FEATS].std(dim=0, keepdim=True).clamp_min(1e-6)
        neuron_feats = neuron_feats.clone()
        neuron_feats[:, :self.PER_NEURON_FEATS] = (
            neuron_feats[:, :self.PER_NEURON_FEATS] - mu
        ) / sd

        global_feat = torch.tensor([
            self.prev_ce - self.orig_ce,
            self.fraction_pruned,
            self.prev_acc,
        ], device=self.device, dtype=torch.float32)

        self._alive_map = alive_map
        return neuron_feats, global_feat, alive_map
