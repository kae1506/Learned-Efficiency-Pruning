# Crisp Findings

Terse index of what we know, what we're answering, what's next. Each finding is one statement + one or two lines of proof + intuition. Full detail per finding lives in its appendix below; original full reasoning in [claude-notes.md](claude-notes.md).

---

## Findings

**F1 — Weights alone encode redundancy.** A BiLSTM reading only weight matrices prunes \~60–79% of an MNIST MLP at\<0.1–3.7pp drop. No data, no gradients needed.

**F2 — pg19's zero-shot CE screening numbers (50K-token sample) were unrepresentative; the real gap survives at proper sample size.** `zero_shot_ce_check.py`'s original 50K-token check gave GPT-2 CE=3.326 (ppl 27.83), OPT-125M CE=2.910 (ppl 18.35). Re-run at 245K tokens (matching the sweep scripts' own `n_test_tokens` default): GPT-2 CE=3.9255 (ppl 50.68), OPT-125M CE=3.4508 (ppl 31.53) — both roughly doubled, confirming pg19's book-to-book heterogeneity makes a small deterministic-streamed sample unrepresentative, not a GPT-2-specific bug. The B6-relevant part survives: gap = GPT-2_CE − OPT_CE was +0.416 nats at 50K, +0.4746 nats at 245K — same direction, similar (slightly larger) magnitude. Treat 27.83/18.35 as superseded; 50.68/31.53 are the correct zero-shot pg19 baselines to compare sweep results against.
