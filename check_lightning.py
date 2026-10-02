"""Prompt 1 check: Lightning Strike hits player, hearts decrease, episode terminates.
Run with: uv run python check_lightning.py
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS, BOSS_STATES, BOSS_ATTACKS
from boss_env.env import _BS_START, _BA_START

MAX_STEPS = 2000   # well above 90s / (4/60s) = 1350 decisions

env = BossFightEnv(weapon="katana", fight=1)
obs, _ = env.reset()

print(f"{'step':>5}  {'px':>7}  {'bx':>7}  {'dist':>6}  "
      f"{'hearts':>6}  {'boss_state':>12}  {'boss_attack':>15}  {'windup':>7}")
print("-" * 80)

prev_hearts = env.player.hearts
hits = 0

for step in range(MAX_STEPS):
    mask   = env.action_masks()
    valid  = [a for a in Action if mask[a]]
    action = random.choice(valid)

    obs, reward, terminated, truncated, _ = env.step(action)

    px = env.player.x
    bx = env.boss.x
    dist = abs(bx - px)
    hearts = env.player.hearts
    bs_hot = obs[_BS_START : _BS_START + len(BOSS_STATES)]
    ba_hot = obs[_BA_START : _BA_START + len(BOSS_ATTACKS)]
    bs = BOSS_STATES[bs_hot.argmax()]
    ba = BOSS_ATTACKS[ba_hot.argmax()]
    windup = obs[OBS_SCALARS.index("boss_windup_timer")]

    if hearts < prev_hearts:
        hits += 1
        print(f"{step:>5}  {px:>7.1f}  {bx:>7.1f}  {dist:>6.1f}  "
              f"{hearts:>6}  {bs:>12}  {ba:>15}  {windup:>7.3f}   <<< HIT")
        prev_hearts = hearts
    elif step % 100 == 0:
        print(f"{step:>5}  {px:>7.1f}  {bx:>7.1f}  {dist:>6.1f}  "
              f"{hearts:>6}  {bs:>12}  {ba:>15}  {windup:>7.3f}")

    if terminated:
        print(f"\n[TERMINATED at step {step}] hearts={hearts}, hits={hits}")
        break
    if truncated:
        print(f"\n[TRUNCATED at step {step}] hearts={hearts} (timeout)")
        break
else:
    print(f"\n[loop ended without termination] hits={hits}")

# Assertions
assert hits > 0, "No hits landed – Lightning Strike never fired"
assert terminated or truncated, "Episode never ended"
if terminated:
    assert env.player.hearts <= 0 or env.boss.hp <= 0, "Terminated but neither player nor boss died"

print("All assertions passed (OK)")
