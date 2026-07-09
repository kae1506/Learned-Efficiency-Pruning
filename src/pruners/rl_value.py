import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """
    State-value function V(s) for the prune-MDP.

    The state is variable-length: at any timestep the alive-neuron set has
    a different size. To produce a single scalar V(s), this net:
      1. Encodes each alive neuron's features through a shared MLP.
      2. Mean-pools across alive neurons to a fixed-size vector.
      3. Concatenates the global trajectory features.
      4. Passes through a head MLP -> scalar.

    Used by the PPO trainer to compute advantages via GAE, replacing the
    EMA scalar baseline used in REINFORCE.
    """

    def __init__(self, feat_dim: int, global_dim: int, hidden: int = 64):
        super().__init__()
        self.neuron_enc = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + global_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, neuron_feats: torch.Tensor, global_feat: torch.Tensor) -> torch.Tensor:
        # neuron_feats: [N_alive, feat_dim]
        # global_feat:  [global_dim]
        h = self.neuron_enc(neuron_feats)            # [N_alive, hidden]
        pooled = h.mean(dim=0)                       # [hidden]
        x = torch.cat([pooled, global_feat], dim=-1) # [hidden + global_dim]
        return self.head(x).squeeze(-1)              # scalar
