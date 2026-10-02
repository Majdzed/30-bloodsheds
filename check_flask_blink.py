"""Check script for Flask healing and Boss Blink teleportation.
Run with: uv run python check_flask_blink.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Flask Drinking, healing, and masking
# ============================================================
print("=== Test 1: Flask Drinking and Healing ===")
env = BossFightEnv(weapon="katana", fight=1)
env.reset()
env.player.flasks = 2  # Give 2 flasks for testing
env.boss.state = BossState.STUN
env.boss.action_timer = 9999  # Stun boss so it doesn't attack during flask test

p = env.player
assert p.flasks == 2
assert p.hearts == p.max_hearts

# Full hearts: USE_FLASK should be masked
mask = env.action_masks()
assert not mask[Action.USE_FLASK], "USE_FLASK should be masked when at full health"

# Damage player
p.hearts = 2
mask = env.action_masks()
assert mask[Action.USE_FLASK], "USE_FLASK should be legal when damaged and having flasks"

# Drink flask
obs, reward, term, trunc, _ = env.step(Action.USE_FLASK)
assert p.state == PlayerState.DRINK or p.hearts == 4  # May finish if frame_skip spans duration

# Let drink complete
for _ in range(15):
    if p.state != PlayerState.DRINK:
        break
    env.step(Action.IDLE)

assert p.state == PlayerState.IDLE, "Player did not return to IDLE after drinking"
assert p.flasks == 1, f"Expected 1 flask remaining, got {p.flasks}"
assert p.hearts == 4, f"Expected 4 hearts after healing 2 hearts, got {p.hearts}"
print(f"  Healed to {p.hearts} hearts. Flasks remaining: {p.flasks}")

# Consume last flask
p.hearts = 1
env.boss.x = 750.0
env.step(Action.USE_FLASK)
for _ in range(15):
    if p.state != PlayerState.DRINK:
        break
    env.step(Action.IDLE)

assert p.flasks == 0, f"Expected 0 flasks remaining, got {p.flasks}"
assert p.hearts == 3, f"Expected 3 hearts after healing, got {p.hearts}"

# Empty flasks: USE_FLASK should be masked even when damaged
mask = env.action_masks()
assert not mask[Action.USE_FLASK], "USE_FLASK should be masked when 0 flasks remain"
print("  Flask healing and masking verified (OK)\n")

# ============================================================
# Test 2: Boss Blink Teleport with Lightning Strike
# ============================================================
print("=== Test 2: Boss Blink Teleportation ===")
env_blink = BossFightEnv(weapon="katana", fight=1)
env_blink.reset()

# Position player and boss far apart (> 150 u)
env_blink.player.x = 100.0
env_blink.boss.x = 700.0
assert env_blink.boss.blink_cd == 0, "Blink cooldown should be 0 on reset"

# Let boss decide ability at distance
blink_triggered = False
teleported = False

for step in range(30):
    b = env_blink.boss
    p = env_blink.player
    dist_before = abs(b.x - p.x)

    obs, reward, term, trunc, _ = env_blink.step(Action.IDLE)

    dist_after = abs(b.x - p.x)
    if b.current_ability_id == 1 and b.blink_cd > 0:
        blink_triggered = True

    # Check if teleport happened (sudden jump from >150u to ~80u)
    if dist_after <= 100.0 and dist_before > 150.0:
        teleported = True
        print(f"  Boss blinked! Distance changed from {dist_before:.1f} u to {dist_after:.1f} u at step {step}")
        break

assert blink_triggered, "Boss did not trigger blink with Lightning Strike at distance"
assert teleported, "Boss did not teleport to within 100 u during slash windup"
assert env_blink.boss.blink_cd > 0, "Blink cooldown was not set"

# Verify observation reflects blink cooldown
obs_blink = obs[OBS_SCALARS.index("boss_blink_cd")]
assert obs_blink > 0.0, f"obs boss_blink_cd should be > 0, got {obs_blink}"
print(f"  boss_blink_cd observation: {obs_blink:.3f}")
print("  Boss blink mechanics and cooldown verified (OK)\n")

print("All assertions passed (OK)")
