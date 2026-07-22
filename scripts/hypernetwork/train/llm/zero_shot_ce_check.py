"""
Quick GO/NO-GO check for B6 (diary/ideas.md): before committing to a full
λ-sweep training run, just measure GPT-2 small's and OPT-125M's zero-shot
CE on pg19 (no training, no pruning, no gates -- pure frozen-model eval).

Hypothesis being screened: GPT-2 is zero-shot WORSE (higher CE) than
OPT-125M on pg19 (long-form books -- OPT trained on BookCorpus+Gutenberg,
GPT-2's WebText has no book-length content at all), mirroring the gap that
made OPT-125M's WikiText-2 pruning sweep show an in-domain improvement
(F19) while GPT-2's did not (F18). If the gap doesn't show up here in the
predicted direction, the B6 sweep isn't worth running on this dataset.

Each model evaluated under its OWN established sliding-window protocol
(same conventions as train_pruner_gpt2.py / train_pruner_opt125m.py):
  GPT-2:    max_length=1024, stride=512
  OPT-125M: max_length=2048, stride=1024

Small token budget by default (--n_tokens, default 50,000) so this runs in
a few minutes on a laptop CPU -- this is a screening check, not the final
measurement. Bump it up if the result is close and you want more precision
before deciding.

Usage:
    python zero_shot_ce_check.py [--n_tokens 50000] [--device cpu]
"""
import argparse
import contextlib
import itertools

import numpy as np
import torch
from tqdm import tqdm


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate_ce(model, ids_flat, device, max_length: int, stride: int, desc: str) -> float:
    """Identical sliding-window protocol used throughout this project's
    train_pruner_*.py scripts -- no token double-counted, near-full context
    for every scored token past the first window."""
    total_len = ids_flat.size(0)
    total_nll = total_tokens = 0
    prev_end = 0
    positions = list(range(0, total_len, stride))
    for begin in tqdm(positions, desc=desc, unit="window", leave=False, dynamic_ncols=True):
        end = min(begin + max_length, total_len)
        trg_len = end - prev_end
        ids = ids_flat[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100
        with autocast_ctx(device):
            loss = model(ids, labels=labels).loss
        n_tok = (labels[:, 1:] != -100).sum().item()
        total_nll += loss.item() * n_tok
        total_tokens += n_tok
        prev_end = end
        if end == total_len:
            break
    return total_nll / total_tokens


def get_pg19_ids(tokenizer, n_tokens: int) -> torch.Tensor:
    from datasets import load_dataset
    # Plain "pg19"/"deepmind/pg19" use a legacy dataset-loading-script format
    # the current `datasets` library no longer supports (verified locally --
    # both raise "Dataset scripts are no longer supported"). This parquet
    # mirror of the same official test split works with streaming=True.
    ds = load_dataset("emozilla/pg19-test", split="test", streaming=True)
    ids = []
    for example in tqdm(ds, desc="streaming pg19", unit="book"):
        ids.extend(tokenizer(example["text"])["input_ids"])
        if len(ids) >= n_tokens:
            break
    if len(ids) < n_tokens:
        raise RuntimeError(f"pg19 test stream exhausted at {len(ids)} tokens, wanted {n_tokens}.")
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_tokens", type=int, default=50_000,
                    help="Screening check -- small on purpose. Raise this if the "
                         "result is close and you want a more precise number.")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    # ---- GPT-2 small ----
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    print("Loading GPT-2 small ...", flush=True)
    gpt2 = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    gpt2.eval()
    for p in gpt2.parameters():
        p.requires_grad_(False)
    gpt2_tok = GPT2TokenizerFast.from_pretrained("gpt2")

    print(f"Streaming pg19 for GPT-2's tokenizer ({args.n_tokens:,} tokens) ...", flush=True)
    gpt2_ids = get_pg19_ids(gpt2_tok, args.n_tokens)
    gpt2_ce = evaluate_ce(gpt2, gpt2_ids, device, max_length=1024, stride=512, desc="GPT-2 on pg19")
    print(f"GPT-2 small  pg19 CE = {gpt2_ce:.4f} nats  (ppl = {np.exp(gpt2_ce):.3f})\n")

    del gpt2
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- OPT-125M ----
    from transformers import OPTForCausalLM, AutoTokenizer
    print("Loading OPT-125M ...", flush=True)
    opt = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m", use_safetensors=True, torch_dtype=torch.float32
    ).to(device)
    opt.eval()
    for p in opt.parameters():
        p.requires_grad_(False)
    opt_tok = AutoTokenizer.from_pretrained("facebook/opt-125m")

    print(f"Streaming pg19 for OPT's tokenizer ({args.n_tokens:,} tokens) ...", flush=True)
    opt_ids = get_pg19_ids(opt_tok, args.n_tokens)
    opt_ce = evaluate_ce(opt, opt_ids, device, max_length=2048, stride=1024, desc="OPT-125M on pg19")
    print(f"OPT-125M     pg19 CE = {opt_ce:.4f} nats  (ppl = {np.exp(opt_ce):.3f})\n")

    # ---- verdict ----
    gap = gpt2_ce - opt_ce   # positive = GPT-2 worse, the predicted direction
    print("=" * 70)
    print(f"GPT-2 CE = {gpt2_ce:.4f}   OPT-125M CE = {opt_ce:.4f}   gap = {gap:+.4f} nats")
    if gap > 0.05:
        print(f"GPT-2 IS worse on pg19 (predicted direction, gap={gap:+.4f} nats, "
              f"ppl ratio {np.exp(gap):.2f}x) -- B6's calibration-gap hypothesis has a "
              f"real gap to test here. Worth running the full sweep.")
    elif gap > -0.05:
        print(f"Gap is near zero ({gap:+.4f} nats) -- no meaningful calibration gap on "
              f"pg19 in either direction. Not a good B6 candidate; try a different dataset "
              f"(OpenSubtitles, raw Reddit) before committing to a sweep here.")
    else:
        print(f"GPT-2 is actually BETTER on pg19 ({gap:+.4f} nats, wrong direction) -- "
              f"contradicts the prediction. Do not run the sweep on this dataset.")
    print("=" * 70)


if __name__ == "__main__":
    main()
