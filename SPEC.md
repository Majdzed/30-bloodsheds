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
