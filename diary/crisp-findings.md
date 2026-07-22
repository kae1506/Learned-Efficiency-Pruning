# Crisp Findings

Terse index of what we know, what we're answering, what's next. Each finding is one statement + one or two lines of proof + intuition. Full detail per finding lives in its appendix below; original full reasoning in [claude-notes.md](claude-notes.md).

---

## Findings

**F1 — Weights alone encode redundancy.** A BiLSTM reading only weight matrices prunes \~60–79% of an MNIST MLP at\<0.1–3.7pp drop. No data, no gradients needed.
