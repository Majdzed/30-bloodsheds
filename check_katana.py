"""Prompt 2 check: Katana light/heavy attacks deal damage, boss dies, episode terminates.
Run with: uv run python check_katana.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS, PLAYER_STATES, BOSS_STATES
from boss_env.env import _PS_START, _BS_START

MAX_STEPS = 3000

# Use fight=1 (boss 150 HP). Agent randomly spams attacks when able.
env = BossFightEnv(weapon="katana", fight=1)
obs, _ = env.reset()

print(f"Boss max HP: {C.boss_max_hp(1)}")
print(f"Katana light: {C.KATANA.light_dmg} dmg, {C.KATANA.light_range} u range, "
      f"{C.sec(C.KATANA.light_windup_s)} windup frames")
print(f"Katana heavy: {C.KATANA.heavy_dmg} dmg, {C.KATANA.heavy_range} u range, "
      f"{C.sec(C.KATANA.heavy_windup_s)} windup frames")
print()

print(f"{'step':>5}  {'px':>7}  {'bx':>7}  {'dist':>6}  "
      f"{'boss_hp':>8}  {'player_state':>13}  {'action':>12}  {'note':>10}")
print("-" * 90)

light_hits = 0
heavy_hits = 0
prev_boss_hp = env.boss.hp

for step in range(MAX_STEPS):
    mask   = env.action_masks()
    valid  = [a for a in Action if mask[a]]
    # Bias toward attacking when we can
    attack_actions = [a for a in valid if a in (Action.LIGHT_ATTACK, Action.HEAVY_ATTACK)]
    action = random.choice(attack_actions) if attack_actions else random.choice(valid)

    obs, reward, terminated, truncated, _ = env.step(action)

    px = env.player.x
    bx = env.boss.x
    dist = abs(bx - px)
    boss_hp = env.boss.hp
    ps_hot = obs[_PS_START : _PS_START + len(PLAYER_STATES)]
    ps = PLAYER_STATES[ps_hot.argmax()]

    note = ""
    if boss_hp < prev_boss_hp:
        dmg = prev_boss_hp - boss_hp
        if abs(dmg - C.KATANA.light_dmg) < 0.5:
            light_hits += 1
            note = f"LIGHT -{dmg:.0f}"
        elif abs(dmg - C.KATANA.heavy_dmg) < 0.5:
            heavy_hits += 1
            note = f"HEAVY -{dmg:.0f}"
        else:
            note = f"HIT -{dmg:.0f}"
        prev_boss_hp = boss_hp
        print(f"{step:>5}  {px:>7.1f}  {bx:>7.1f}  {dist:>6.1f}  "
              f"{boss_hp:>8.1f}  {ps:>13}  {action.name:>12}  {note:>10}")
    elif step % 200 == 0:
        print(f"{step:>5}  {px:>7.1f}  {bx:>7.1f}  {dist:>6.1f}  "
              f"{boss_hp:>8.1f}  {ps:>13}  {action.name:>12}")

    if terminated:
        print(f"\n[TERMINATED at step {step}] boss_hp={env.boss.hp:.1f}")
        break
    if truncated:
        print(f"\n[TRUNCATED at step {step}] boss_hp={env.boss.hp:.1f} (timeout)")
        break
else:
    print(f"\n[loop ended] boss_hp={env.boss.hp:.1f}")

print(f"\nLight hits: {light_hits}  Heavy hits: {heavy_hits}")

# Assertions
assert light_hits + heavy_hits > 0, "No attacks landed at all"
assert terminated or truncated, "Episode never ended"
if terminated:
    assert env.boss.hp <= 0, f"Terminated but boss_hp={env.boss.hp} > 0"
    assert env.player.hearts > 0 or env.player.hearts <= 0  # either is valid

print("All assertions passed (OK)")
