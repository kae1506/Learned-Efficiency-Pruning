"""
Smoke-test: BiLSTM pruner pipeline on frozen GPT-2 small.

Verifies the full pipeline end-to-end in just a few steps:
  1. Download pretrained GPT-2 small weights (no GPT-2 training)
  2. Extract the 12 FFN c_fc weight matrices
  3. Build a BiLSTM pruner (same arch as MNIST pruner, different layer shapes)
  4. Train pruner for n_steps:
       loss = (lm_loss_masked - lm_loss_orig) + λ * avg_gate

Run from project root:
    venv/bin/python scripts/hypernetwork/train/train_pruner_gpt2_smoke.py
    venv/bin/python scripts/hypernetwork/train/train_pruner_gpt2_smoke.py --n_steps 5
"""

import sys
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

sys.path.append(".")
from src.pruners.bilstm import Pruner

# GPT-2 small architecture constants
N_LAYERS = 12    # number of transformer blocks
FFN_OUT  = 3072  # FFN intermediate width (c_fc output features)
FFN_IN   = 768   # model hidden dim (c_fc input features)


def get_ffn_weights(model: GPT2LMHeadModel) -> list[torch.Tensor]:
    """
    Extract c_fc weight matrices for all 12 FFN blocks, detached.

    HuggingFace GPT-2 uses Conv1D (not nn.Linear): weight is stored
    as [in, out] = [768, 3072], the TRANSPOSE of the standard convention.
    We transpose to [out, in] = [3072, 768] so each row is one FFN neuron's
    incoming weight vector — matching the BiLSTM row_encoder interface.

    Returns: list of 12 tensors, each [3072, 768], requires_grad=False.
    """
    return [
        block.mlp.c_fc.weight.T.detach()  # [768, 3072].T → [3072, 768]
        for block in model.transformer.h
    ]


def make_gate_hook(gate: torch.Tensor):
    """
    Forward hook for block.mlp.act (the post-c_fc GELU activation).

    Multiplies the post-GELU tensor by the gate vector, zeroing out pruned
    FFN neurons before they reach c_proj. Gate is a [3072] STE binary tensor,
    so autograd can flow gradients back to the pruner through this multiply.
    """
    def hook(module, input, output):
        # output: [B, T, 3072] — post-GELU hidden state
        # gate:   [3072]       — broadcasts over B and T
        return output * gate.view(1, 1, -1)
    return hook


def pruner_step(
    pruner: Pruner,
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    sparsity_weight: float,
) -> dict:
    """
    One BiLSTM pruner training step on GPT-2.

    loss = (lm_loss_masked - lm_loss_orig) + λ * avg_gate

    lm_loss_orig:   frozen GPT-2 LM loss (scalar baseline, no grad needed)
    lm_loss_masked: same pass but FFN neurons are gated off by the pruner
    sparsity term:  λ * mean(gate) pushes the pruner to shut more neurons off

    Gradient only flows to pruner params — GPT-2 weights stay frozen.
    """
    optimizer.zero_grad()

    # -- Baseline LM loss: frozen GPT-2, no hooks, no grad --
    # Wrapped in no_grad because we only need the scalar value as a baseline.
    with torch.no_grad():
        orig_loss = model(input_ids, labels=input_ids).loss.item()

    # -- Get gates from BiLSTM pruner (STE: hard binary forward, soft backward) --
    weight_matrices = get_ffn_weights(model)  # 12 × [3072, 768]
    gates = pruner(weight_matrices)           # 12 × [3072] gate tensors

    # -- Register activation hooks to apply gates during the masked forward --
    # Hook sits after GELU, before c_proj: zeros out pruned neurons.
    hooks = []
    for block, gate in zip(model.transformer.h, gates):
        h = block.mlp.act.register_forward_hook(make_gate_hook(gate))
        hooks.append(h)

    # -- Masked forward: grad flows gate → activation → loss → pruner params --
    masked_loss = model(input_ids, labels=input_ids).loss

    # Remove hooks immediately; don't leave them dangling between steps.
    for h in hooks:
        h.remove()

    # -- Pruner loss: penalise accuracy drop + reward sparsity --
    avg_gate = torch.stack([g.mean() for g in gates]).mean()
    loss = (masked_loss - orig_loss) + sparsity_weight * avg_gate

    loss.backward()
    optimizer.step()

    return {
        "loss":        loss.item(),
        "orig_lm":     orig_loss,
        "masked_lm":   masked_loss.item(),
        "avg_gate":    avg_gate.item(),
        "pct_pruned":  (1.0 - avg_gate.item()) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-2 BiLSTM pruner smoke test")
    parser.add_argument("--n_steps",         type=int,   default=2,
                        help="number of pruner training steps (default: 2)")
    parser.add_argument("--seq_len",         type=int,   default=32,
                        help="token sequence length per step (default: 32)")
    parser.add_argument("--sparsity_weight", type=float, default=0.05,
                        help="lambda: sparsity penalty weight (default: 0.05)")
    parser.add_argument("--lr",              type=float, default=1e-3,
                        help="Adam lr for pruner (default: 1e-3)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -- Load GPT-2 small: downloads ~500MB on first run, cached afterward --
    print("Loading GPT-2 small (download on first run)...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)  # GPT-2 is fully frozen; only pruner params are updated
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GPT-2 loaded: {n_params/1e6:.1f}M params (all frozen)")

    # -- Build BiLSTM pruner for GPT-2's 12 FFN layers --
    # Each layer: 3072 prunable neurons, each described by a 768-dim weight row.
    # This is the same Pruner class used for MNIST — only layer_shapes changes.
    layer_shapes = [(FFN_OUT, FFN_IN)] * N_LAYERS  # 12 × (3072, 768)
    pruner = Pruner(layer_shapes, embed_dim=64, lstm_hidden=64).to(device)
    optimizer = torch.optim.Adam(pruner.parameters(), lr=args.lr)

    n_pruner_params = sum(p.numel() for p in pruner.parameters())
    total_prunable  = N_LAYERS * FFN_OUT
    print(f"Pruner: {n_pruner_params:,} params")
    print(f"Pruning target: {N_LAYERS} FFN layers × {FFN_OUT} neurons = {total_prunable:,} total prunable neurons")
    print()

    # -- Input: random tokens for smoke test (replace with real data for full run) --
    # A real training run should use a WikiText / OpenWebText DataLoader here.
    torch.manual_seed(42)
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)

    # -- Pruner training loop --
    print(f"{'Step':>4}  {'loss':>8}  {'orig_lm':>8}  {'masked_lm':>9}  {'avg_gate':>8}  {'pruned%':>7}")
    print("-" * 60)

    for step in range(1, args.n_steps + 1):
        m = pruner_step(pruner, model, optimizer, input_ids, args.sparsity_weight)
        print(
            f"{step:>4}  "
            f"{m['loss']:>+8.4f}  "
            f"{m['orig_lm']:>8.4f}  "
            f"{m['masked_lm']:>9.4f}  "
            f"{m['avg_gate']:>8.4f}  "
            f"{m['pct_pruned']:>6.1f}%"
        )

    print()
    print("Smoke test passed — pipeline is end-to-end OK.")
    print("Next: swap random input_ids for a real DataLoader, set n_steps=12500.")


if __name__ == "__main__":
    main()
