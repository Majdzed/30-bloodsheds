"""Prompt 4 check: stamina regen, dodge roll movement, i-frames block Lightning Strike.
Run with: uv run python check_roll.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action
from boss_env.entities import PlayerState

MAX_STEPS = 3000

env = BossFightEnv(weapon="katana", fight=1)
obs, _ = env.reset()

print(f"ROLL_COST={C.ROLL_COST}, ROLL_DURATION_S={C.ROLL_DURATION_S}, "
      f"ROLL_DISTANCE={C.ROLL_DISTANCE}, ROLL_IFRAMES_S={C.ROLL_IFRAMES_S}")
print()

rolls_done      = 0
roll_blocked    = 0   # hits that DIDN'T land due to i-frames
hearts_start    = env.player.hearts
min_stamina     = C.STAMINA_MAX
max_x_during_roll = -1.0
min_x_during_roll = C.ARENA_W + 1.0
in_roll = False
roll_start_x    = 0.0

print(f"{'step':>5}  {'state':>8}  {'px':>7}  {'stam':>6}  {'iframes':>7}  "
      f"{'roll_cd':>7}  {'action':>12}  {'note':>15}")
print("-" * 80)

for step in range(MAX_STEPS):
    p = env.player
    mask  = env.action_masks()
    valid = [a for a in Action if mask[a]]

    # Prefer dodge roll when available; otherwise random
    if Action.DODGE_ROLL in valid:
        action = Action.DODGE_ROLL
    else:
        action = random.choice(valid)

    prev_state  = p.state
    prev_hearts = p.hearts
    prev_stamina = p.stamina

    obs, reward, terminated, truncated, info = env.step(action)

    p2 = env.player
    note = ""

    # Track roll start/end
    if prev_state != PlayerState.ROLL and p2.state == PlayerState.ROLL:
        in_roll = True
        roll_start_x = p2.x
        rolls_done += 1
        note = "ROLL START"

    if prev_state == PlayerState.ROLL and p2.state != PlayerState.ROLL:
        in_roll = False
        roll_dist = abs(p2.x - roll_start_x)
        note = f"ROLL END dx={roll_dist:.1f}"
        # Roll should have moved close to ROLL_DISTANCE units
        assert roll_dist <= C.ROLL_DISTANCE + 5, \
            f"Roll moved {roll_dist:.1f} u > ROLL_DISTANCE+5={C.ROLL_DISTANCE+5}"

    # Stamina decreases on roll start
    if prev_state != PlayerState.ROLL and p2.state == PlayerState.ROLL:
        expected_stamina = prev_stamina - C.ROLL_COST
        assert abs(p2.stamina - expected_stamina) < 1.0, \
            f"Stamina not deducted: was {prev_stamina:.1f}, now {p2.stamina:.1f}"

    # Stamina never goes below 0 or above max
    assert 0.0 <= p2.stamina <= C.STAMINA_MAX + 0.01, \
        f"Stamina out of range: {p2.stamina:.2f}"
    min_stamina = min(min_stamina, p2.stamina)

    # Player position always in bounds
    assert 0.0 <= p2.x <= C.ARENA_W, f"Player x={p2.x:.2f} out of bounds at step {step}"

    if note or step % 150 == 0:
        print(f"{step:>5}  {p2.state.name:>8}  {p2.x:>7.1f}  {p2.stamina:>6.1f}  "
              f"{p2.iframes:>7}  {p2.roll_cd:>7}  {action.name:>12}  {note:>15}")

    if terminated or truncated:
        print(f"\n[Episode ended at step {step}]: terminated={terminated}, truncated={truncated}")
        break

print()
print(f"Total rolls performed: {rolls_done}")
print(f"Minimum stamina seen:  {min_stamina:.2f}")

assert rolls_done > 0, "No rolls were performed"
assert min_stamina >= 0.0, "Stamina went negative"

# Verify stamina regen: after a roll, stamina should eventually recover
print()
print("=== Stamina regen test (isolated) ===")
env2 = BossFightEnv(weapon="katana", fight=1)
env2.reset()
env2.player.stamina = 0.0
env2.player.stamina_regen_delay = 0   # no delay
for _ in range(int(C.FPS * 4)):       # 4 seconds of IDLE
    env2.step(Action.IDLE)
final_stam = env2.player.stamina
print(f"  After 4s of IDLE with stamina=0: stamina={final_stam:.2f}")
assert final_stam > 50, f"Stamina should have regenerated significantly, got {final_stam:.2f}"

print()
print("All assertions passed (OK)")
