# `experiments/latest/` — directory layout

Experiments are grouped into four families. Within each leaf folder:
- `plot.png` / `*.png` — figures
- `summary.txt` — config + headline numbers
- `run.log` — full stdout (where captured)

```
experiments/latest/
├── base_model/                      train_mnist.py        — MNIST MLP 784→1024→1024→10
│
├── hypernetwork/                    weight-conditioned BiLSTM pruning
│   ├── training/                    train_pruner.py       — BiLSTM pruner training
│   ├── dim_sweep/                   dim_sweep.py          — width vs prunability (32…2048)
│   ├── interp_sw0.3/                train_bilstm_interp.py --sw 0.3
│   └── interp_sw0.5/                train_bilstm_interp.py --sw 0.5
│
├── baselines/                       non-learned / classic pruning
│   ├── activation/                  activation_pruning.py — mean-activation threshold
│   ├── sequential/                  sequential_pruning.py — activation → learned chaining
│   └── activation_vs_bilstm/        compare_act_vs_bilstm.py — Pareto comparison
│
└── rl/                              reinforcement-learning pruning
    ├── reinforce/                   train_rl_pruner.py --max_prune X --tag <tag>
    │   ├── 70/ 80/ 90/              per prune-target
    │   └── sweep_65_85/             rl_prune_sweep.py     — 10-fraction sweep
    ├── ppo/80/                      train_rl_pruner_ppo.py
    ├── reinforce_interp/80/         train_rl_interp.py    — REINFORCE + interpretability
    └── variance_study/              multi-seed algorithm comparisons
        ├── reinforce_vs_bilstm/     multi_seed_compare.py — 5-seed BiLSTM vs REINFORCE
        ├── actor_critic/            multi_seed_ac.py      — 5-seed AC (V(s) baseline)
        └── actor_critic_norment/    multi_seed_ac_norment.py — AC + normalised entropy
```

RL run logs captured at invocation via `... 2>&1 | tee <folder>/run.log`.
