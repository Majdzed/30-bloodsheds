# 30 Bloodsheds: Boss Gauntlet RL

**Course:** Reinforcement Learning and Optimal Control (2026/2027)  
**Program:** 3CS – Artificial Intelligence & Data Science  
**Instructors:** N. Farhi & M. Bouali  
**Project:** Development of a Game-Based Environment for Reinforcement Learning  

---

## 1. Project Description

**30 Bloodsheds** is a 2D boss battle combat game designed as a Deep Reinforcement Learning (DRL) testbed. The agent must master real-time combat mechanics against an escalating 30-fight boss gauntlet.

The game is modeled as a discrete-time Markov Decision Process (MDP) with a 60 FPS internal simulation tick and a 4-frame action repeat ($\Delta t = 1/15\text{s}$ per decision step). The player chooses between two distinct weapons:
- **Katana**: High-risk, high-DPS close-quarters melee combat requiring tight parry windows and punish timing.
- **Revolver**: Tactical ranged combat with bullet management (6-round cylinder), reload vulnerabilities, and piercing heavy shots.

Across the 30-stage gauntlet, the boss unlocks **7 deadly abilities** with escalating stats, hazard density, and multi-ability overlaps:
1. Melee Slash / Frenzy (close-range punish)
2. Dark Spikes (ground hazard tracking the player's position)
3. Dark Wave (jump-only ground-sweeping wave)
4. Light Bolt (parry/roll fast projectile)
5. Siphon Heal (spawns 3 orbiting heal spheres that the player must destroy)
6. Ultimate Dark Orb (arena-wide shockwave; **must be parried**)
7. Dual-Blade Frenzy (relentless 3-hit combo)

The game provides two operational modes:
- **Interactive Mode**: Graphical Pygame interface for human play (`play.py`).
- **Headless Mode**: High-throughput simulation (~5,000–10,000 steps/sec) with zero graphics overhead for training and evaluation (`boss_env/env.py`).

---

## 2. Working Environment & Machine Specifications

### Hardware Specifications
- **Operating System:** Windows 11 Pro 64-bit (Build 10.0.22631)
- **CPU:** AMD64 Family 25 (6 Physical Cores / 12 Logical Threads)
- **System Memory (RAM):** 24 GB (23.86 GB usable)
- **Acceleration:** Headless CPU vectorization (PyTorch CPU / OpenMP)

### Main Python Packages & Versions
The project follows the standard `uv` package manager format (`pyproject.toml`, `uv.lock`):

| Package | Version | Purpose |
|:---|:---|:---|
| **Python** | `3.12.7` | Runtime environment |
| **Gymnasium** | `1.0.0` | RL environment standard interface |
| **Stable-Baselines3** | `2.9.0` | Deep RL baseline algorithms |
| **SB3-Contrib** | `2.9.0` | `MaskablePPO` implementation with action masking |
| **PyTorch** | `2.14.1+cpu` | Neural network computation and autograd |
| **Pygame** | `2.6.1` | 2D game rendering and human interactive loop |
| **Pandas** | `2.2.3` | Benchmark metrics logging and tabular aggregation |
| **Matplotlib** | `3.10.0` | Visual plotting of rewards and benchmark curves |
| **TensorBoard** | `2.18.0` | Real-time training convergence logging |
| **Jupyter / Notebook** | `7.3.2` | Interactive report visualization (`evaluate.ipynb`) |

---

## 3. Project Structure

```
Project - 30 bloodshed/
├── pyproject.toml              # uv project configuration and dependency constraints
├── uv.lock                     # Deterministic dependency lockfile
├── README.md                   # Setup, execution guide, and environment description
├── SPEC.md                     # Comprehensive technical specifications & MDP definition
│
├── train.py                    # Trains RL agent with MaskablePPO & curriculum learning
├── test.py                     # Evaluates RL agent vs. Rule-Based baseline (CSVs output)
├── play.py                     # Interactive mode: lets a human play the game with GUI
├── observe.py                  # Visualizes a single episode played by a trained RL/baseline agent
├── evaluate.ipynb              # Displays benchmark tables, convergence, and comparative plots
├── plot_results.py             # Generates standalone publication-ready figures
│
├── boss_env/                   # Core environment package
│   ├── __init__.py
│   ├── env.py                  # Headless Gymnasium BossFightEnv (state machine & kinematics)
│   ├── entities.py             # Player and Boss dataclasses and states
│   ├── abilities.py            # Boss abilities definition and gauntlet unlock schedules
│   ├── constants.py            # Global physics, hitbox, weapon, and damage constants
│   ├── networks.py             # Custom 4-stream neural network architecture
│   └── renderer.py             # Pygame drawing module (HUD, animations, particles, effects)
│
├── agents/                     # Agent implementations
│   ├── __init__.py
│   └── rule_based.py           # Expert baseline state-machine agent for both weapons
│
├── models/                     # Saved model checkpoints (*.zip)
│   ├── curriculum_katana_gauntlet_final.zip
│   └── curriculum_revolver_gauntlet_final.zip
│
├── runs/                       # TensorBoard convergence logs & Monitor CSVs
│   ├── katana_curriculum_.../
│   └── revolver_curriculum_.../
│
├── results/                    # Benchmark evaluation results (CSV reports)
│   ├── eval_report_1790962623.csv  # Katana benchmark (F1 - F30)
│   └── eval_report_1791204312.csv  # Revolver benchmark (F1 - F30)
│
└── plots/                      # Rendered figures & performance comparisons
    ├── training_curves.png
    ├── benchmark_comparison.png
    └── weapon_comparison.png
```

---

## 4. Setup and Installation

The environment is managed using [uv](https://docs.astral.sh/uv/) for instant, reproducible installation.

### 1. Clone & Synchronize Environment
```bash
# Clone the repository and enter directory
cd "Project - 30 bloodshed"

# Synchronize all dependencies from lockfile
uv sync
```

---

## 5. How to Run Each File

### 1. Train the RL Agent (`train.py`)
Trains an agent using `MaskablePPO` with action masking, our custom 4-stream feature extractor, and curriculum replay:
```bash
# Train Katana agent across the 8 gauntlet curriculum stages (240k steps)
uv run python train.py --weapon katana --curriculum --use-custom-extractor --stages 1 4 8 12 16 21 26 30 --timesteps-per-stage 30000

# Train Revolver agent across the gauntlet stages
uv run python train.py --weapon revolver --curriculum --use-custom-extractor --stages 1 4 8 12 16 21 26 30 --timesteps-per-stage 30000

# Train on a single specific fight (e.g. Fight 1)
uv run python train.py --weapon katana --fight 1 --total-timesteps 50000
```
*TensorBoard logs are saved to `runs/` and model checkpoints to `models/`.*

### 2. Evaluate Agents & Generate CSV Reports (`test.py`)
Runs rigorous multi-episode benchmarks comparing the trained MaskablePPO agent against the Rule-Based baseline:
```bash
# Evaluate Katana RL vs Rule-Based across gauntlet stages + 10-run endurance test
uv run python test.py --fights 1 4 8 12 16 21 26 30 --episodes 15 --weapon katana --model models/curriculum_katana_gauntlet_final.zip --gauntlet-runs 10

# Evaluate Revolver RL vs Rule-Based
uv run python test.py --fights 1 4 8 12 16 21 26 30 --episodes 15 --weapon revolver --model models/curriculum_revolver_gauntlet_final.zip --gauntlet-runs 10
```
*Evaluation results are saved as structured CSV files in `results/`.*

### 3. Play the Game Interactively (`play.py`)
Allows a human player to play in interactive mode with real-time Pygame graphics:
```bash
# Play Fight 1 with Katana
uv run python play.py --weapon katana --fight 1

# Play Fight 8 with Revolver
uv run python play.py --weapon revolver --fight 8
```
**Keyboard Controls:**
- `A` / `D`: Move Left / Right
- `W` / `Space`: Jump
- `J`: Light Attack (Slash / Pistol Shot)
- `K`: Heavy Attack (Spin Slash / Piercing Shot)
- `L`: Roll / Dodge (Invulnerability frames)
- `U`: Parry / Block (Nullifies attack & staggers boss)
- `I`: Super Move (Blood Katana / Ultra Plasma)
- `H`: Use Healing Flask
- `R`: Restart Fight | `Esc`: Quit

### 4. Observe Trained Agent Simulation (`observe.py`)
Visualizes a single episode played in real-time Pygame rendering by the trained RL agent or Rule-Based baseline:
```bash
# Watch the trained Revolver RL agent on Fight 1
uv run python observe.py --agent rl --weapon revolver --model models/curriculum_revolver_gauntlet_final.zip --fight 1

# Watch the Katana RL agent on Fight 8
uv run python observe.py --agent rl --weapon katana --model models/curriculum_katana_gauntlet_final.zip --fight 8

# Watch the Rule-Based baseline on Fight 4
uv run python observe.py --agent rule_based --weapon katana --fight 4
```

### 5. View Evaluation Tables and Plots (`evaluate.ipynb`)
Launches the interactive Jupyter Notebook containing all convergence curves, metrics justifications, interactive Pandas tables, and comparative charts:
```bash
uv run jupyter notebook evaluate.ipynb
```

### 6. View Live Training Convergence in TensorBoard
```bash
uv run tensorboard --logdir runs
```

---

## 6. MDP Formulation & Custom Neural Architecture

### Markov Decision Process (MDP)
- **State Space ($\mathcal{S} \subset \mathbb{R}^{52}$):**
  - Scalar features $[0:32]$: Player/Boss spatial coords ($x, y, v_x, v_y$), relative distance, health fractions, stamina, weapon ammo, reload progress, cooldown timers, active projectile/hazard vectors.
  - One-hot state encodings $[32:52]$: Player state (7), Boss state (5), Boss active attack (8).
- **Action Space ($\mathcal{A}$ - Discrete 10):**
  - $\{0: \text{Move Left}, 1: \text{Move Right}, 2: \text{Jump}, 3: \text{Light Attack}, 4: \text{Heavy Attack}, 5: \text{Roll}, 6: \text{Parry}, 7: \text{Super}, 8: \text{Flask}, 9: \text{Idle}\}$.
- **Action Masking:**
  - Prevents illegal transitions (e.g. attacking or parrying while airborne or rolling, shooting with 0 ammo).

### Custom 4-Stream Neural Architecture (`boss_env/networks.py`)
Per the project guidelines requiring a **customized neural network architecture rather than a predefined MLP**, we implemented `BossGauntletFeaturesExtractor`:
```
Obs Box(52,)
 ├── Player Stream  [obs: 0-14, 32-39] ──> Linear(64) ──> LayerNorm ──> ReLU ──┐
 ├── Boss Stream    [obs: 14-19, 39-52] ─> Linear(64) ──> LayerNorm ──> ReLU ──┼──> Concat(192) ──> Linear(128) ──> ReLU
 ├── Hazard Stream  [obs: 19-24] ────────> Linear(32) ──> LayerNorm ──> ReLU ──┤                      (Fused Representation)
 └── Context Stream [obs: 24-32] ────────> Linear(32) ──> LayerNorm ──> ReLU ──┘
```
This inductive bias separates self-kinematics from opponent state and arena hazards, enabling rapid convergence across curriculum transitions.

---

## 7. Benchmark Results Summary

### Per-Fight Win Rates (15 Episodes per Stage)

| Gauntlet Stage | Boss HP | Active Abilities | Katana Baseline | Katana MaskablePPO | Revolver Baseline | Revolver MaskablePPO |
|:---|:---:|:---|:---:|:---:|:---:|:---:|
| **Fight 1** | 150 | Melee Slash | 100% | **100% (Flawless)** | 100% | **100% (Flawless)** |
| **Fight 4** | 193 | + Light Bolt | 100% | **100%** | 100% | **100% (Flawless)** |
| **Fight 8** | 250 | + Dark Spikes | 0% | **100%** | 60% | **93.3%** |
| **Fight 12**| 307 | + Dark Wave | 20% | **86.7%** | 60% | **80.0%** |
| **Fight 16**| 365 | + Siphon Heal | 0% | **100%** | 13.3% | **93.3%** |
| **Fight 21**| 437 | + Ultimate Dark Orb | 0% | **100%** | 6.7% | **66.7%** |
| **Fight 26**| 509 | + Dual-Blade Frenzy | 0% | **60.0%** | 6.7% | 0.0% |
| **Fight 30**| 566 | All 7 Abilities Maxed | 0% | **66.7%** | 0.0% | 0.0% |

### Gauntlet Endurance Benchmark (10 Consecutive Clears)
- **Katana MaskablePPO:** Average **7.7 / 30**, Maximum **30 / 30 (Full Gauntlet Clear)**.
- **Revolver MaskablePPO:** Average **9.6 / 30**, Maximum **12 / 30**.
- **Rule-Based Baseline:** Katana average 7.0 / 30 (Max 8); Revolver average 7.8 / 30 (Max 9).

---

## 8. References & Citations

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. Huang, S., & Ontañón, S. (2022). *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*. The International FLAIRS Conference Proceedings.
3. Raffin, A. et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. Journal of Machine Learning Research (JMLR).
4. Towers, M. et al. (2023). *Gymnasium: A Standard Interface for Reinforcement Learning Environments*.
5. Game design and combat mechanics inspired by *Sekiro: Shadows Die Twice* (FromSoftware) parry-stagger dynamics and *Enter the Gungeon* (Dodge Roll) tactical projectile spacing.
