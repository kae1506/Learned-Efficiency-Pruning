import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    """
    Permutation-invariant policy over the alive-neuron set.

    Each alive neuron j gets a feature vector phi(j). A small encoder produces a
    key per neuron; a global query is computed from trajectory features
    (CE gap, fraction pruned, current accuracy). Logits = keys . query.
    """

    def __init__(self, feat_dim: int, global_dim: int, hidden: int = 64):
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

    def forward(self, neuron_feats: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # neuron_feats: [N_alive, feat_dim]
        # global_feat:  [global_dim]
        keys = self.key_enc(neuron_feats)            # [N_alive, hidden]
        query = self.query_enc(global_feat)          # [hidden]
        return keys @ query                          # [N_alive]
