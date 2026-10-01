# Boss Gauntlet RL

A 30-fight boss gauntlet used as a Deep Reinforcement Learning environment.
Full rules and the MDP definition are in [SPEC.md](SPEC.md).

## Setup

```bash
uv sync
```

## Run

```bash
uv run python train.py --weapon katana     # train the RL agent
uv run python test.py                      # evaluate RL vs rule-based agent
uv run python play.py                      # play it yourself
uv run python observe.py                   # watch one episode
uv run jupyter notebook evaluate.ipynb     # plots and tables
uv run tensorboard --logdir runs           # training curves
```

## Structure

| Path | Content |
|---|---|
| `boss_env/constants.py` | numeric game parameters |
| `boss_env/abilities.py` | boss ability table and unlock schedule |
| `boss_env/env.py` | headless Gymnasium environment (no drawing) |
| `boss_env/renderer.py` | Pygame drawing only |
| `boss_env/networks.py` | custom features extractor |
| `models/`, `runs/`, `results/`, `plots/` | saved models, TensorBoard logs, CSVs, figures |

(Report sections on machine specs and package versions: fill in later.)
