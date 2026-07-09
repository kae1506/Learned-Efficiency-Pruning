import torch
import torch.nn as nn


class BernoulliPolicyNet(nn.Module):
    """
    Per-neuron INDEPENDENT Bernoulli prune policy.

    Same key/query backbone as PolicyNet, but instead of a softmax over the
    alive set ("pick exactly k"), each alive neuron j gets an independent
    prune probability:

        p_j = sigmoid( key_j · query + bias )

    The action is one independent Bernoulli draw per neuron. Its joint
    log-probability factorises EXACTLY:

        log π(a|s) = Σ_j [ b_j log p_j + (1-b_j) log(1-p_j) ]

    so there is no without-replacement joint-logprob proxy (the bias in the
    multinomial "pick-k" action), and each neuron gets clean per-neuron credit.
    The number pruned per step is variable and learned.

    Initialisation: the query encoder's final layer is zero-init so query≈0 at
    step 0, making every logit ≈ `init_bias` → uniform p ≈ sigmoid(init_bias).
    With init_bias=-4.8, p≈0.0081, so expected pruned per step ≈ 0.0081·N_alive
    ≈ 16 at N=2048 — matching the k=16 chunk regime so episode length starts
    comparable. The policy then learns to raise/lower individual neurons' p_j.
    """

    def __init__(self, feat_dim: int, global_dim: int, hidden: int = 64,
                 init_bias: float = -4.8):
        super().__init__()
        self.key_enc = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.query_enc = nn.Sequential(
            nn.Linear(global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        # zero-init query head → query≈0 → logit≈bias (uniform low p) at start
        nn.init.zeros_(self.query_enc[-1].weight)
        nn.init.zeros_(self.query_enc[-1].bias)
        self.bias = nn.Parameter(torch.tensor(float(init_bias)))

    def forward(self, neuron_feats: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        keys  = self.key_enc(neuron_feats)        # [N_alive, hidden]
        query = self.query_enc(global_feat)       # [hidden]
        return keys @ query + self.bias           # [N_alive] raw logits (pre-sigmoid)
