# Boss Gauntlet: Game and MDP Specification (v1.0)

Single source of truth for the game rules and the RL formulation. Code constants live in `boss_env/constants.py` and `boss_env/abilities.py`. If you change a number, change it here and there.

Values marked *(proposed)* were not part of the original design and are playtest-tunable. Everything else was agreed by the team.

## 1. Concept

- 1v1 side-view boss fight in an old city under moonlight. Two characters: the player and the boss.
- The player fights the same boss **30 times** (a gauntlet). The boss gains new abilities as fights progress and its HP scales up.
- **RL agent = the player.** The boss is scripted. The agent keeps its weights across fights (curriculum).
- Two weapon loadouts, one per run: **Katana** or **Revolver**. No switching in combat.

## 2. World and timing

- Arena: 800 units wide, movement on the x axis only (plus jump on y).
- 60 FPS simulation tick. The agent acts every **4 frames** (frame skip, 15 decisions/s) and repeats the action in between. *(proposed, configurable)*
- Both characters face each other automatically, so there is no facing state. *(proposed)*
- Fight time limit: **90 s**. Timeout counts as a loss. *(proposed)*

## 3. Player

| Property | Value |
|---|---|
| Hearts | 4 (max 5 after the fight-5 upgrade) |
| Move speed | 220 u/s *(proposed)* |
| Stamina | max 100, regen 25/s after a 0.5 s delay |
| Dodge roll | costs 30 stamina (3 rolls in a row max), 0.35 s, 120 u, i-frames for the first 0.2 s, cooldown 0.6 s *(proposed)* |
| Jump | 0.6 s airtime, 100 u apex, no horizontal steering, no air attacks |
| Parry | window 12 frames (0.2 s), cooldown 1.0 s |
| Flask | heals 2 hearts, drinking takes 0.6 s and leaves the player vulnerable *(proposed)* |

Attack, jump and parry cost no stamina.

**Roll** i-frames avoid every boss hit except the Ultimate Dark Orb.
**Jump** clears ground hits (Dark Spikes, Dark Wave). It does not avoid Lightning Strike, Light Bolt, Dual-Blade Frenzy or the Dark Orb.
**Parry** negates the next incoming hit inside its window: Lightning Strike, Dark Wave, Light Bolt, Dual-Blade Frenzy (one strike per parry), Ultimate Dark Orb. It cannot parry Dark Spikes or Siphon Heal. A successful parry gives **+1 super stack** and stuns the boss for 0.5 s if the parried attack was melee. *(stun proposed)*
No post-hit invulnerability, so a full Dual-Blade Frenzy can deal 3 hearts.

### Flasks and upgrades

- Fights 1 to 3: 1 flask. From fight 4: 2 flasks.
- After fight 5: one upgrade, either **+1 max heart** or **+1 flask**. This is a per-run config (`upgrade = health | flask`), not an agent decision.
- Hearts, flasks and the super meter reset at the start of every fight. *(assumed)*

## 4. Weapons

| | Katana | Revolver |
|---|---|---|
| Light attack | 15 dmg, 0.20 s windup, 120 u range | 6 dmg, 0.15 s windup, full arena, uses 1 bullet |
| Heavy attack | 35 dmg, 0.50 s windup, 150 u range | 18 dmg, 0.60 s windup, full arena, piercing, uses 2 bullets |
| Recovery after attack | 0.40 s | 0.15 s light, 0.25 s heavy *(proposed)* |
| Ammo | n/a | 6 bullets |
| Super (3 parries) | **Blood Katana**: 0.8 s lock, 3 slashes of 30 dmg = 90, 200 u reach | **Ultra Plasma Shot**: 0.4 s cast, fullscreen laser, 80 dmg, staggers boss 1.0 s *(stagger proposed)*, clears all boss projectiles |

- Damage lands at the end of the windup. Melee is an instant range check toward the boss.
- The player is invulnerable during a super. *(proposed)*
- Using a super consumes all 3 stacks.
- The boss has no armor or blocking in v1.

### Revolver reload

- Full reload takes **1.2 s** (72 frames). It starts automatically when the magazine is empty, or after 0.5 s idle. *(proposed)*
- **Parry, jump and shooting cancel it** and reset its progress. Dodge roll and movement do not. *(shooting cancel proposed)*

## 5. Boss

### HP scaling

Linear between anchors: fight 1 = 150, fight 15 = 350, fight 30 = 600 (rounded to an integer).

### Movement (scripted)

- Walks toward the player at 120 u/s when idle and out of attack range.
- **Lightning Strike is decoupled from the blink:**
  - Melee slash: 2.0 s cooldown, used whenever the player is within 100 u.
  - Blink: only if the player is more than 150 u away and the blink's own 8.0 s cooldown is ready. The boss teleports to within 100 u of the player during the slash windup. Otherwise it just walks up and slashes.

### Abilities

| ID | Ability | Windup | Active | Damage | Range / size | Cooldown | Unlock fight |
|---|---|---|---|---|---|---|---|
| 1 | Lightning Strike | 0.50 s | 0.15 s | 1 heart | 120 u arc | 2.0 s | 1 |
| 2 | Dark Spikes | 0.80 s warning | 0.25 s | 1 heart | 140 u circle at the player's x when cast *(diameter assumed)* | 5.0 s | 8 |
| 3 | Dark Wave | 0.60 s | travels at 280 u/s | 1 heart | ground projectile | 4.0 s | 12 |
| 4 | Light Bolt | 0.30 s | travels at 600 u/s | 1 heart | high projectile | 2.0 s | 4 |
| 5 | Siphon Heal | 0.75 s cast | up to 3.0 s | none | self | 14.0 s | 21 |
| 6 | Ultimate Dark Orb | 1.20 s charge | 2.0 s sweep (400 u/s) | 3 hearts (0 if parried) | arena-height, cannot be dodged | 18.0 s | 26 |
| 7 | Dual-Blade Frenzy | 0.25 s flash | 3 strikes x 0.1 s | 1 heart per hit | 150 u dash | 3.5 s | 16 |

### Siphon Heal (orb mechanic)

- On channel start, **4 orbs** spawn around the boss (within about 120 u of it). The boss stays in place.
- The boss consumes one orb every 0.75 s, each giving **+15 HP** (60 HP total, equal to 20 HP/s over 3 s).
- Any attack hit destroys an orb (1 hit). The Revolver's piercing heavy can destroy several in one shot.
- If all remaining orbs are destroyed, the channel is interrupted and the boss is stunned for 1.0 s. *(stun proposed)*
- The heal that lands equals 15 HP per orb that was consumed.

### Boss ability selection (scripted) *(proposed)*

When the boss is idle and its 0.3 s global recovery is over:

1. If Siphon Heal is unlocked, ready, and boss HP is below 60%, cast it.
2. Otherwise, build the candidate list from unlocked abilities that are off cooldown and valid for the distance:
   - Lightning Strike: player within 100 u (or beyond 150 u with the blink ready)
   - Dual-Blade Frenzy: player between 100 u and 300 u
   - Dark Wave and Light Bolt: player beyond 150 u
   - Dark Spikes and Ultimate Dark Orb: any distance
3. Pick one uniformly at random (seeded RNG). If none is valid, walk toward the player.

## 6. MDP

### Episode

- One episode = one fight. The 30-fight gauntlet is the curriculum.
- `terminated`: boss HP <= 0 (win) or player hearts <= 0 (loss).
- `truncated`: 90 s time limit (counted as a loss in the reward).

### Action space: Discrete(10) with action masking

| ID | Action | Masked when |
|---|---|---|
| 0 | IDLE | never |
| 1 | MOVE_LEFT | at left wall, or not actionable |
| 2 | MOVE_RIGHT | at right wall, or not actionable |
| 3 | JUMP | airborne, or not actionable |
| 4 | LIGHT_ATTACK | in recovery, Revolver empty, or not actionable |
| 5 | HEAVY_ATTACK | in recovery, Revolver below 2 bullets, or not actionable |
| 6 | DODGE_ROLL | stamina below 30, on cooldown, or not actionable |
| 7 | PARRY | on cooldown, or not actionable |
| 8 | USE_FLASK | no flasks left, hearts full, or not actionable |
| 9 | USE_SUPER | super meter below 3 stacks, or not actionable |

"Not actionable" = rolling, stunned, drinking, mid-attack, or in a super.
Masking needs `MaskablePPO` from `sb3-contrib`. The standard SB3 PPO does not support masks.

### Observation space: Box(shape=(52,), float32)

Scalars (32), in this order:

| Group | Features |
|---|---|
| Player (14) | `x_rel` (boss minus player, /800), `player_x`, `player_y`, `player_vy`, `player_hp` (hearts/5), `player_stamina`, `weapon_type` (0 Katana, 1 Revolver), `ammo_level`, `reload_progress`, `super_meter` (0 to 1), `flask_count` (/3), `roll_cd`, `parry_cd`, `parry_active` |
| Boss (5) | `boss_hp` (hp/max_hp), `boss_windup_timer`, `boss_is_healing`, `heal_orbs_remaining` (/4), `boss_blink_cd` |
| Hazards (5) | `spike_rel_x`, `spike_timer`, `proj_dark_rel_x`, `proj_light_rel_x`, `orb_parry_window` (1.0 when the orb's impact is within the next 12 frames) |
| Context (8) | `fight_index` (/30), 7-bit `unlocked_ability` mask |

One-hot features (20): `player_state` (7: idle, attack, roll, parry, drink, stun, super), `boss_state` (5: idle, windup, attack, heal, stun), `boss_attack` (8: none + 7 abilities).

Context features are for the agent only and are **not drawn in the human HUD**.

### Reward (identical in all 30 fights)

| Event | Reward |
|---|---|
| Damage dealt to the boss | +1.0 x (damage / boss max HP) |
| Heart lost | -0.25 each |
| Successful parry | +0.3 |
| Heal orb destroyed | +0.1 each |
| Heal fully interrupted | +0.3 bonus |
| Win | +1.0 |
| Loss or timeout | -1.0 |
| Each decision step | -0.001 |

Report convergence with the **undiscounted total reward per episode** in TensorBoard, plus a sliding-window average.

## 7. Modes

- **Headless**: no window, used by `train.py` and `test.py`.
- **Interactive**: Pygame window for `play.py` and `observe.py`.
- **Human lose modes**: retry the whole 30-fight gauntlet, or only the latest fight. For RL the episode is always one fight.
- **Evaluation** also reports how many consecutive gauntlet fights each agent clears.

## 8. Training plan

- Two separate training runs: Katana and Revolver.
- Weights carry over from fight to fight, with some episodes replayed from earlier unlock stages to limit forgetting.
- Algorithm: MaskablePPO with a custom features extractor.
- Baseline: rule-based agent, one version per weapon, evaluated on the same fights and seeds.

## 9. Adaptive Boss (Interactive / Human Play Mode Only)

> **Scope:** This feature is active **only** in `play.py` when a human is playing (not for RL training or evaluation).  The flag is `env._adaptive_boss = False` by default and is flipped by `env.set_player_model(model)`.  All constants below are tunable in `boss_env/adaptation.py`.

### 9.1 Concept

The boss maintains a **Markov-style player model** that accumulates observations of how the human player responds to each boss ability.  After collecting enough data, the boss skews its random ability selection to exploit the player's tendencies, making it progressively harder to exploit predictable patterns.

### 9.2 Player Model (`boss_env/adaptation.py`)

**Context** = `(boss_ability_id ∈ {1..7},  distance_bucket ∈ {close, mid, far})`
- `close`: distance < 100 u
- `mid`: 100 u ≤ distance ≤ 300 u
- `far`: distance > 300 u

**Response** = dominant player behaviour observed during the boss ability's windup + active window, collapsed to one label (priority order):
`parry > roll > jump > attack > move_toward > move_away > idle`

**Count table**: `counts[context][response]` — integer observation counts.

**Tendencies** (queried each time the boss picks an ability):
```
tendency[r] = Σ_ctx counts[ctx][r] / total_samples
```
Returns uniform `1/7` per response if no data yet.

### 9.3 Counter Matrix

Derived from SPEC §3 (player evasion rules) and §5 (ability table).  Each ability has a weight per response:

| Response     | Best countering abilities |
|:---|:---|
| `roll`       | Dark Orb (6) — roll i-frames do NOT avoid it |
| `jump`       | Lightning Strike (1), Light Bolt (4), Frenzy (7), Dark Orb (6) |
| `parry`      | Dark Spikes (2) — cannot be parried |
| `idle/stand` | Dark Spikes (2), Lightning Strike (1) |
| `move_away`  | Dark Wave (3), Light Bolt (4), blink+Lightning Strike (1) |
| `move_toward`| Lightning Strike (1) — melee range punishment |

### 9.4 Weighted Ability Selection

Replaces the uniform `np_random.choice(candidates)` in the boss IDLE decision:

```
alpha       = min(ALPHA_MAX, total_samples / SAMPLES_REF)
learned_p   = softmax(counter_score)        # via tendency dot-product
final_p     = (1 - alpha) * uniform + alpha * learned_p
chosen_id   = weighted_sample(candidates, final_p, rng=self.np_random)
```

**Constants** (playtest-tunable):

| Constant          | Value | Purpose |
|:---|:---:|:---|
| `ALPHA_MAX`       | 0.6   | Max boss-learns fraction; keeps the boss beatable |
| `SAMPLES_REF`     | 50    | Observations needed to reach `ALPHA_MAX` |
| `LEARN_UNTIL_FIGHT` | 10  | Model freezes (no new obs accepted) after this fight |
| `APPLY_FROM_FIGHT`  | 4   | Weighted selection starts here (fights 1-3: only 1 candidate) |

The `Siphon Heal` priority rule (SPEC §5 step 1) is **unchanged** and still executes before any weighted selection.

### 9.5 Lifetime and Persistence

- The `PlayerModel` is created once per game session in `play.py`.
- It **persists across fights and retries** within a session.
- It is **saved to `player_model.json`** next to `play.py` when a fight ends and on exit.
- It is **loaded on startup** if the file exists.
- Delete `player_model.json` to start fresh.
- The model is **never attached** when an RL agent is playing (`Agent vs Boss` mode).

### 9.6 Human-Readable Feedback

`PlayerModel.top_tendencies_text()` returns up to 3 lines such as:
```
"You roll 62% of the time"
"You jump 18% of the time"
"You parry 11% of the time"
```
These are printed to the console on exit and can be shown in the post-fight HUD.

### 9.7 Hard Constraints (not to be changed)

- `adaptive_boss=False` by default.  RL training (`train.py`), evaluation (`test.py`), and agent observation (`observe.py`) must **never** call `set_player_model()`.
- The observation space, action space, and reward function are **not modified**.
- No saved models require retraining.
- The boss's seeded RNG (`self.np_random`) is used for the weighted sample so that replay with the same seed + same model state is still reproducible.

## 10. Boss Dodge & Unified Boss Mode (Human Mode Only)

> **Scope:** The Boss Dodge ability is part of the unified `boss_mode = "human"` setting. It is active **always** in `play.py` during "Player vs Boss" mode. "Agent vs Boss" mode as well as `train.py`, `test.py`, and `observe.py` always use `boss_mode = "benchmark"`, keeping the benchmark MDP completely untouched.

### 10.1 Unified Boss Mode

Instead of fragmented toggles, `BossFightEnv` exposes:
- `boss_mode = "benchmark"` (default): The strict, scripted, frozen gauntlet that RL agents train on and are evaluated against. Also used by "Agent vs Boss" mode in `play.py`.
- `boss_mode = "human"`: Always active during "Player vs Boss" mode in `play.py`. Enables both the online Markov adaptive player model (§9) and the evasive Boss Dodge (§10).

### 10.2 Dodge Trigger & Scaling Rules

- **Unlock:** Available starting from **Fight 10** (`DODGE_UNLOCK_FIGHT = 10`).
- **State Requirement:** Boss can only initiate a dodge from `BossState.IDLE` with `recovery_timer == 0` and `dodge_cd == 0`. It cannot dodge while winding up, attacking, healing, stunned, or recovering.
- **Evaluation Timing:** When a player attack has been in windup for at least `BOSS_REACT_TIME` and has not yet landed, the boss rolls the seeded RNG against `DODGE_PROB`. Evaluated at most once per player attack.
- **Supers:** Blood Katana and Ultra Plasma Shot **cannot be dodged**.

**Dynamic Scaling Schedules (Fights 10 to 30):**

| Parameter | Fight 10 | Fight 30 | Formula (Linear) |
|:---|:---:|:---:|:---|
| `BOSS_REACT_TIME` | 0.20 s | 0.10 s | `0.20 - (fight - 10) * 0.005` |
| `DODGE_PROB` | 0.50 | 0.80 | `0.50 + (fight - 10) * 0.015` |
| `DODGE_COOLDOWN` | 5.0 s | 3.5 s | `5.0 - (fight - 10) * 0.075` |
| Max Dodges / 10s | 2 | 2 | Rate-limited rolling window |

*Note on light attacks:* At Fight 10, Katana light windup is 0.20s and Revolver light windup is 0.15s. Because react time is 0.20s, light attacks naturally land before the react threshold, guaranteeing that light attacks cannot be dodged at default react time. Heavy attacks (0.50s / 0.60s windup) give the boss ample time to react.

### 10.3 Dodge Movement & Invulnerability

- **Invulnerability:** 0.25 s (15 frames) of active i-frames (`DODGE_IFRAMES_S`). Hits landing during i-frames deal 0 damage and are consumed.
- **Movement:** Moves 120 u directly away from the player (`DODGE_DISTANCE = 120.0`), clamped to the arena boundaries `[0, ARENA_W]`.
- **Cornered Escape:** If the boss has $< 120\,\text{u}$ of wall space behind it, it executes a cross-dash past the player to the opposite side (`p.x - 120u` or `p.x + 120u`), escaping the corner.
- **Recovery:** Followed by 0.30 s (18 frames) of vulnerable recovery (`DODGE_RECOVERY_S`) where the boss cannot act. Any hit landing during recovery deals full damage.

### 10.4 Observation Space Mapping & Known Limitation

- To preserve strict compatibility with SB3 policies and existing checkpoints, the 52-float observation space shape is **unchanged**.
- `BossState.DODGE` (state enum value 5) is mapped to `BossState.IDLE` in the 5-element one-hot boss state vector:
  ```python
  obs_boss_state = BossState.IDLE if b.state == BossState.DODGE else b.state
  obs[_BS_START + int(obs_boss_state)] = 1.0
  ```
- *Known Limitation:* An RL agent observing human mode would observe the boss as `IDLE` during a dodge. Since RL agents only evaluate in `benchmark` mode, this has zero impact on training or benchmarking.
