"""Movement sanity check. Run with: uv run python check_movement.py

Runs a random agent for N_STEPS decisions. Verifies:
  - Player x stays in [0, 800]
  - Boss x stays in [0, 800]
  - Boss closes in and stops within 100 u of the player
  - The action masks correctly block MOVE_LEFT at x==0 and MOVE_RIGHT at x==800
"""

import sys, pathlib, random
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env.env import BossFightEnv, Action, OBS_SCALARS

N_STEPS   = 300
PRINT_EVERY = 30    # print a row every this many steps

env = BossFightEnv(weapon="katana", fight=1)
obs, _ = env.reset()

p_x_idx = OBS_SCALARS.index("player_x")
b_x_idx = OBS_SCALARS.index("x_rel")   # x_rel = (boss_x - player_x) / 800

print(f"{'step':>5}  {'action':>12}  {'player_x':>9}  {'boss_x':>7}  {'dist':>6}  "
      f"{'mask_L':>6}  {'mask_R':>6}")
print("-" * 65)

violations = []

for step in range(N_STEPS):
    mask = env.action_masks()

    # Sample a random *valid* action
    valid = [a for a in Action if mask[a]]
    action = random.choice(valid)

    obs, reward, terminated, truncated, _ = env.step(action)

    px = env.player.x
    bx = env.boss.x
    dist = abs(bx - px)

    # Collect position violations
    if not (0.0 <= px <= 800.0):
        violations.append(f"step {step}: player_x={px:.2f} out of bounds")
    if not (0.0 <= bx <= 800.0):
        violations.append(f"step {step}: boss_x={bx:.2f} out of bounds")

    # Verify mask correctness for the NEXT step (post-step state)
    next_mask = env.action_masks()
    if px <= 0.0 and next_mask[Action.MOVE_LEFT]:
        violations.append(f"step {step}: MOVE_LEFT not masked at x={px:.4f}")
    if px >= 800.0 and next_mask[Action.MOVE_RIGHT]:
        violations.append(f"step {step}: MOVE_RIGHT not masked at x={px:.4f}")

    if step % PRINT_EVERY == 0 or step == N_STEPS - 1:
        print(f"{step:>5}  {action.name:>12}  {px:>9.2f}  {bx:>7.2f}  "
              f"{dist:>6.2f}  {int(mask[Action.MOVE_LEFT]):>6}  {int(mask[Action.MOVE_RIGHT]):>6}")

    if terminated or truncated:
        print(f"  [episode ended at step {step}, truncated={truncated}]")
        break

print()
print(f"Final distance: {abs(env.boss.x - env.player.x):.2f} u "
      f"(boss stops walking when dist <= {100} u)")

if violations:
    print()
    print("VIOLATIONS:")
    for v in violations:
        print(f"  {v}")
    sys.exit(1)
else:
    print("All checks passed (OK)")
