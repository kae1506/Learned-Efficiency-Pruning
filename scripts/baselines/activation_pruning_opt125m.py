"""
Activation-based pruning baseline for OPT-125M FFN neurons, matched to a
target sparsity level for direct comparison against the trained BiLSTM
pruner (see train_pruner_opt125m.py). Standalone, no repo deps.

Algorithm (same convention as scripts/baselines/activation_pruning.py,
extended from a fixed threshold to a target-sparsity threshold):
  1. Run the frozen model over N calibration batches of WikiText-2 train
     data, recording each FFN neuron's mean post-ReLU activation (hooked
     at fc2's forward_pre_hook -- same point apply_gates() uses in the
     trained-pruner script, so both methods gate at an identical position).
  2. Pool all 36,864 neurons' (12 layers x 3072) mean activations into one
     global ranking and pick the threshold that prunes the lowest-magnitude
     TARGET_PRUNE_FRAC of them -- not a fixed literature threshold value,
     since the point here is an apples-to-apples sparsity comparison, not
     reproducing a specific paper's absolute cutoff. The literature part is
     the CRITERION (mean-magnitude activation pruning, e.g. Hu et al. 2016
     APoZ-family methods), not the cutoff value.
  3. Zero those neurons' post-activation output via the same apply_gates()
     hook mechanism as the trained pruner (forward_pre_hook on fc2).
  4. Evaluate on the FULL WikiText-2 test set with the same sliding-window
     protocol (max_length=2048, stride=1024) used to report the trained
     pruner's numbers, so ppl is directly comparable.

TARGET_PRUNE_FRAC = 0.4256 (42.56%) matches the λ=0.75 trained-pruner run's
ACTUAL final sparsity: summary.txt's "per-block neurons kept" list sums to
21175/36864 kept = 42.56% pruned -- this is the mask that produced the
reported pruned ppl=27.086, computed by re-running pruner.eval() once
training finished. (The summary's other headline number, "final avg keep
gate: 0.5524" / "44.76% pruned", is a training-loop snapshot from history[
-1] taken one gradient step earlier than the post-training gate recompute --
not the number tied to the reported ppl. Matching to that would compare
against a sparsity level nothing was actually evaluated at.)
"""

import contextlib
import os
import datetime

import torch
import torch.nn.functional as F
from tqdm import tqdm

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

N_LAYERS = 12
N_INTER  = 3072
TARGET_PRUNE_FRAC = 0.4256   # matches trained pruner's actual λ=0.75 sparsity (21175/36864 kept)
CALIB_BATCHES = 50
BATCH_SIZE = 8
SEQ_LEN = 512
EVAL_MAX_LENGTH = 2048
EVAL_STRIDE = 1024


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_opt125m(device):
    from transformers import OPTForCausalLM
    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m", use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def get_loaders(seq_len: int, batch_size: int, num_workers: int = 0):
    """Identical tokenization to train_pruner_opt125m.py::get_loaders (join-then-tokenize
    fix -- see that file's docstring for why per-line tokenization is wrong for OPT)."""
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    def tokenize_split(split):
        return tokenizer("\n\n".join(raw[split]["text"]))["input_ids"]

    test_ids  = torch.tensor(tokenize_split("test"),  dtype=torch.long)
    train_ids = torch.tensor(tokenize_split("train"), dtype=torch.long)

    total = (train_ids.size(0) // seq_len) * seq_len
    train_blocked = Dataset.from_dict(
        {"input_ids": train_ids[:total].view(-1, seq_len).tolist()}
    )
    train_blocked.set_format(type="torch", columns=["input_ids"])
    train_loader = DataLoader(train_blocked, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    return train_loader, test_ids


@torch.no_grad()
def collect_mean_activations(model, train_loader, n_batches, device):
    """Mean post-ReLU activation per FFN neuron, hooked at fc2's forward_pre_hook
    (post-activation, pre-down-projection) -- same point apply_gates() gates."""
    sums = [torch.zeros(N_INTER, device=device) for _ in range(N_LAYERS)]
    counts = [0] * N_LAYERS

    hooks = []
    def make_hook(idx):
        def hook(module, args):
            # fc2's input rank varies by backend/attention-impl (3D [batch,seq,3072]
            # on some paths, 2D [batch*seq, 3072] flattened on others) -- reshape to
            # 2D explicitly rather than assume a fixed rank.
            x = args[0].reshape(-1, N_INTER)   # [n_tokens, 3072], post-ReLU
            sums[idx] += x.sum(dim=0)
            counts[idx] += x.shape[0]
            return None
        return hook
    for i, block in enumerate(model.model.decoder.layers):
        hooks.append(block.fc2.register_forward_pre_hook(make_hook(i)))

    loader_iter = iter(train_loader)
    for _ in tqdm(range(n_batches), desc="calibration", unit="batch"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)
        ids = batch["input_ids"].to(device)
        with autocast_ctx(device):
            model(ids)

    for h in hooks:
        h.remove()

    return [s / c for s, c in zip(sums, counts)]


def build_global_threshold_gates(mean_acts, target_prune_frac, device):
    """Pool all layers' neuron magnitudes, pick the global percentile cutoff
    that prunes target_prune_frac of the total 36,864 neurons."""
    all_acts = torch.cat(mean_acts)
    n_total = all_acts.numel()
    n_prune = int(round(n_total * target_prune_frac))
    threshold = torch.kthvalue(all_acts, n_prune).values.item()

    gates = [(ma > threshold).float() for ma in mean_acts]
    n_pruned_actual = sum((g == 0).sum().item() for g in gates)
    return gates, threshold, n_pruned_actual, n_total


@contextlib.contextmanager
def apply_gates(model, gates):
    hooks = []
    for block, gate in zip(model.model.decoder.layers, gates):
        def make_hook(g):
            def hook(module, args):
                x = args[0]
                # rank-agnostic broadcast: (1,)*  (x.dim()-1) + (-1,), matches
                # x's actual rank (2D or 3D depending on backend) instead of
                # assuming 3D.
                view_shape = (1,) * (x.dim() - 1) + (-1,)
                return (x * g.view(*view_shape),)
            return hook
        hooks.append(block.fc2.register_forward_pre_hook(make_hook(gate)))
    try:
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def evaluate(model, test_ids, device, gates=None, desc="eval",
            max_length=EVAL_MAX_LENGTH, stride=EVAL_STRIDE):
    """Same sliding-window CE protocol as train_pruner_opt125m.py::evaluate."""
    total_len = test_ids.size(0)
    total_nll = total_tokens = 0
    prev_end = 0
    positions = list(range(0, total_len, stride))
    for begin in tqdm(positions, desc=desc, unit="window", leave=False, dynamic_ncols=True):
        end = min(begin + max_length, total_len)
        trg_len = end - prev_end
        ids = test_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        with autocast_ctx(device):
            if gates is None:
                loss = model(ids, labels=labels).loss
            else:
                with apply_gates(model, gates):
                    loss = model(ids, labels=labels).loss
        n_tok = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n_tok
        total_tokens += n_tok
        prev_end = end
        if end == total_len:
            break
    return total_nll / total_tokens


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Loading OPT-125M ...", flush=True)
    model = load_opt125m(device)

    print("Loading WikiText-2 ...", flush=True)
    train_loader, test_ids = get_loaders(SEQ_LEN, BATCH_SIZE)
    print(f"test_tokens={test_ids.size(0):,}", flush=True)

    print(f"\nCollecting mean post-ReLU activations over {CALIB_BATCHES} calibration batches "
          f"({CALIB_BATCHES * BATCH_SIZE * SEQ_LEN:,} tokens) ...", flush=True)
    mean_acts = collect_mean_activations(model, train_loader, CALIB_BATCHES, device)
    for i, ma in enumerate(mean_acts):
        print(f"  L{i}: min={ma.min():.4f} mean={ma.mean():.4f} max={ma.max():.4f}")

    gates, threshold, n_pruned, n_total = build_global_threshold_gates(
        mean_acts, TARGET_PRUNE_FRAC, device
    )
    print(f"\nGlobal threshold = {threshold:.6f} -> {n_pruned}/{n_total} "
          f"({n_pruned/n_total*100:.2f}%) neurons pruned "
          f"(target {TARGET_PRUNE_FRAC*100:.2f}%)")
    per_layer_kept = [int(g.sum().item()) for g in gates]
    print(f"per-layer kept: {per_layer_kept}")

    print("\nEvaluating original (unpruned) ...", flush=True)
    orig_ce = evaluate(model, test_ids, device, gates=None, desc="orig")
    print("Evaluating activation-pruned ...", flush=True)
    pruned_ce = evaluate(model, test_ids, device, gates=gates, desc="pruned")

    import math
    orig_ppl = math.exp(orig_ce)
    pruned_ppl = math.exp(pruned_ce)

    print("\n" + "=" * 70)
    print(f"orig ppl   : {orig_ppl:.3f}")
    print(f"pruned ppl : {pruned_ppl:.3f}  ({n_pruned/n_total*100:.2f}% pruned)")
    print(f"ppl change : {pruned_ppl - orig_ppl:+.3f}")
    print("=" * 70)

    out_dir = "experiments/latest/baselines/activation_opt125m"
    os.makedirs(out_dir, exist_ok=True)
    lines = [
        "=" * 60,
        "ACTIVATION PRUNING BASELINE — OPT-125M FFN, WikiText-2",
        f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "ALGORITHM",
        "  Mean post-ReLU activation per FFN neuron (hooked at fc2's",
        "  forward_pre_hook, same point the trained BiLSTM pruner gates),",
        f"  over {CALIB_BATCHES} calibration batches ({CALIB_BATCHES*BATCH_SIZE*SEQ_LEN:,} tokens)",
        "  of WikiText-2 train. Global magnitude threshold picked to match",
        f"  {TARGET_PRUNE_FRAC*100:.2f}% pruned (= trained pruner's actual λ=0.75 sparsity,",
        "  21175/36864 kept, the mask that produced its reported ppl=27.086 —",
        "  NOT its training-loop 'final avg keep gate' snapshot of 44.76%).",
        "",
        "RESULTS",
        f"  neurons pruned : {n_pruned}/{n_total} ({n_pruned/n_total*100:.2f}%)",
        f"  per-layer kept : {per_layer_kept}",
        f"  orig ppl       : {orig_ppl:.3f}",
        f"  pruned ppl     : {pruned_ppl:.3f}",
        f"  ppl change     : {pruned_ppl - orig_ppl:+.3f}",
        "",
        "COMPARISON — trained BiLSTM pruner, λ=0.75, same eval protocol:",
        "  orig ppl   : 23.941",
        "  pruned ppl : 27.086  (42.56% pruned)",
        "  ppl change : +3.145",
        "=" * 60,
    ]
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nSaved -> {out_dir}/summary.txt")


if __name__ == "__main__":
    main()
