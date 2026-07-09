# `scripts/` — experiment entry points

All scripts run **from the project root**:

```
venv/bin/python scripts/<family>/<name>.py [args]
```

(They use `sys.path.append(".")` and write outputs to paths relative to the
root, so the working directory must be the project root.)

| Family | Script | What it does | Output |
|---|---|---|---|
| **base** | `train_mnist.py` | Train MNIST MLP, save checkpoint | `base_model/` |
| **hypernetwork** | `train_pruner.py` | BiLSTM pruner training | `hypernetwork/training/` |
| | `dim_sweep.py` | Width vs prunability | `hypernetwork/dim_sweep/` |
| | `train_bilstm_interp.py --sw X` | BiLSTM + interpretability | `hypernetwork/interp_sw<X>/` |
| | `bilstm_5seed.py` | BiLSTM 5-seed variance | `hypernetwork/bilstm_5seed/` |
| | `wide_model_prune.py` | Wide-model (shape) generalization | `hypernetwork/transfer_wide2048/` |
| **baselines** | `activation_pruning.py` | Mean-activation threshold | `baselines/activation/` |
| | `sequential_pruning.py` | Activation → learned chaining | `baselines/sequential/` |
| | `compare_act_vs_bilstm.py` | Pareto comparison | `baselines/activation_vs_bilstm/` |
| **rl** | `train_rl_pruner.py --max_prune X --tag T` | REINFORCE pruner | `rl/reinforce/<T>/` |
| | `rl_prune_sweep.py` | REINFORCE 65–85% sweep | `rl/reinforce/sweep_65_85/` |
| | `train_rl_pruner_ppo.py` | PPO pruner | `rl/ppo/<tag>/` |
| | `train_rl_interp.py` | REINFORCE + interpretability | `rl/reinforce_interp/80/` |
| | `multi_seed_compare.py` | 5-seed BiLSTM vs REINFORCE | `rl/variance_study/reinforce_vs_bilstm/` |
| | `multi_seed_ac.py` | 5-seed actor-critic | `rl/variance_study/actor_critic/` |
| | `multi_seed_ac_norment.py` | AC + normalised entropy | `rl/variance_study/actor_critic_norment/` |
| | `chunk_sweep.py` | Prune-chunk k sweep | `rl/chunk_sweep/` |
| | `chunk_k8_k16_3seed.py` | k=8 vs k=16, 3-seed | `rl/chunk_sweep/k8_3seed/` |
| | `bernoulli_3seed.py` | Per-neuron Bernoulli action | `rl/bernoulli_3seed/` |

Library code lives in `src/`: core (`model.py`, `dataset.py`, `train.py`,
`prune_train.py`, `interpretability.py`), pruner nets (`src/pruners/`), RL
trainers + env (`src/rl/`).
