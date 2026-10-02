"""Check script for Revolver weapon mechanics: ammo, attacks, piercing, reload, and Ultra Plasma Shot super.
Run with: uv run python check_revolver.py
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action, OBS_SCALARS
from boss_env.entities import PlayerState, BossState

# ============================================================
# Test 1: Ammo Consumption, Full-Arena Range, and Action Masking
# ============================================================
print("=== Test 1: Revolver Ammo & Range ===")
env = BossFightEnv(weapon="revolver", fight=1)
env.reset()

p = env.player
b = env.boss
# Boss far away (x=700, p.x=150, dist=550 > melee range)
b.x = 700.0
b.state = BossState.STUN
b.action_timer = 9999

assert p.ammo == 6, f"Expected 6 bullets, got {p.ammo}"
mask = env.action_masks()
assert mask[Action.LIGHT_ATTACK] and mask[Action.HEAVY_ATTACK]

# Subtest 1A: Light attack at distance 550u (full arena reach)
hp_before = b.hp
env.step(Action.LIGHT_ATTACK)
# Step through windup (0.15s) and recovery (0.15s)
for _ in range(6):
    env.step(Action.IDLE)
    if p.state == PlayerState.IDLE and p.recovery_timer == 0:
        break

assert p.ammo == 5, f"Expected 5 bullets after light attack, got {p.ammo}"
assert b.hp == hp_before - 6.0, f"Expected 6 damage at range 550u, got {hp_before - b.hp}"
print(f"  Light attack dealt 6 dmg at distance 550u. Ammo: {p.ammo} (OK)")

# Subtest 1B: Heavy attack consumes 2 bullets and deals 18 dmg
hp_before = b.hp
env.step(Action.HEAVY_ATTACK)
for _ in range(15):
    env.step(Action.IDLE)
    if p.state == PlayerState.IDLE and p.recovery_timer == 0:
        break

assert p.ammo == 3, f"Expected 3 bullets after heavy attack, got {p.ammo}"
assert b.hp == hp_before - 18.0, f"Expected 18 damage at range 550u, got {hp_before - b.hp}"
print(f"  Heavy attack dealt 18 dmg at distance 550u. Ammo: {p.ammo} (OK)")

# Fire heavy again -> ammo = 1
env.step(Action.HEAVY_ATTACK)
for _ in range(15):
    env.step(Action.IDLE)
    if p.state == PlayerState.IDLE and p.recovery_timer == 0:
        break
assert p.ammo == 1, f"Expected 1 bullet, got {p.ammo}"

# With ammo=1: Light attack is legal, Heavy attack is MASKED
mask = env.action_masks()
assert mask[Action.LIGHT_ATTACK], "Light attack should be legal with 1 bullet"
assert not mask[Action.HEAVY_ATTACK], "Heavy attack should be MASKED with 1 bullet (needs 2)"
print("  Masking with 1 bullet: Light allowed, Heavy masked (OK)")

# Fire last bullet -> ammo = 0
env.step(Action.LIGHT_ATTACK)
for _ in range(6):
    env.step(Action.IDLE)
    if p.state == PlayerState.IDLE and p.recovery_timer == 0:
        break
assert p.ammo == 0, f"Expected 0 bullets, got {p.ammo}"

# With ammo=0: Both attacks are MASKED
mask = env.action_masks()
assert not mask[Action.LIGHT_ATTACK], "Light attack should be masked with 0 bullets"
assert not mask[Action.HEAVY_ATTACK], "Heavy attack should be masked with 0 bullets"
print("  Masking with 0 bullets: Both attacks masked (OK)\n")

# ============================================================
# Test 2: Auto-Reload and Cancel Rules
# ============================================================
print("=== Test 2: Auto-Reload & Cancel Rules ===")
env_reload = BossFightEnv(weapon="revolver", fight=1)
env_reload.reset()
env_reload.boss.state = BossState.STUN
env_reload.boss.action_timer = 9999

# Empty the magazine
env_reload.player.ammo = 0

# Step once to trigger auto-reload
obs, reward, term, trunc, _ = env_reload.step(Action.IDLE)
assert env_reload.player.reload_timer > 0, "Auto-reload did not start when empty"

reload_obs = obs[OBS_SCALARS.index("reload_progress")]
assert reload_obs > 0.0, f"reload_progress should be > 0, got {reload_obs}"
print(f"  Empty magazine started reload immediately: reload_progress={reload_obs:.3f}")

# Subtest 2A: Movement and Roll do NOT cancel reload
env_reload.step(Action.MOVE_LEFT)
assert env_reload.player.reload_timer > 0, "Movement should NOT cancel reload"
env_reload.step(Action.DODGE_ROLL)
assert env_reload.player.reload_timer > 0, "Dodge roll should NOT cancel reload"
print("  Movement and Dodge Roll did not interrupt reload (OK)")

# Subtest 2B: Jump cancels reload
env_reload.step(Action.JUMP)
assert env_reload.player.reload_timer == 0, "Jump should cancel reload and reset progress"
print("  Jump cancelled reload progress (OK)")

# Land and let full reload finish
for _ in range(12):
    env_reload.step(Action.IDLE)

# Start fresh empty reload and let it complete (72 frames = 18 steps)
env_reload.player.ammo = 0
for _ in range(25):
    env_reload.step(Action.IDLE)
    if env_reload.player.ammo == C.REVOLVER_MAG:
        break

assert env_reload.player.ammo == C.REVOLVER_MAG, f"Expected full magazine (6), got {env_reload.player.ammo}"
print(f"  Reload completed successfully: ammo restored to {env_reload.player.ammo} (OK)\n")

# ============================================================
# Test 3: Piercing Heavy Attack Destroys All Heal Orbs
# ============================================================
print("=== Test 3: Piercing Heavy Attack vs Heal Orbs ===")
env_pierce = BossFightEnv(weapon="revolver", fight=21)
env_pierce.reset()
env_pierce.boss.state = BossState.HEAL
env_pierce.boss.action_timer = 9999
env_pierce._heal_orbs = [250.0, 300.0, 350.0, 400.0]
env_pierce.boss.orbs_remaining = 4
env_pierce.player.ammo = 6

# Execute Revolver heavy attack (piercing across full arena)
env_pierce.step(Action.HEAVY_ATTACK)
for _ in range(15):
    env_pierce.step(Action.IDLE)
    if len(env_pierce._heal_orbs) == 0:
        break

assert len(env_pierce._heal_orbs) == 0, f"Piercing heavy should destroy all heal orbs in line, remaining: {len(env_pierce._heal_orbs)}"
assert env_pierce.boss.state == BossState.STUN, "Interrupting all orbs should stun boss"
print("  Piercing Heavy destroyed all 4 heal orbs in single shot and stunned boss (OK)\n")

# ============================================================
# Test 4: Super: Ultra Plasma Shot
# ============================================================
print("=== Test 4: Ultra Plasma Shot Super ===")
env_super = BossFightEnv(weapon="revolver", fight=4)
env_super.reset()
env_super.boss.state = BossState.IDLE
env_super.boss.x = 650.0
env_super.player.super_stacks = 3

# Spawn incoming boss projectiles
env_super._light_bolts.append({"x": 400.0, "dir": -1.0, "hit": False})
env_super._dark_waves.append({"x": 450.0, "dir": -1.0, "hit": False})
env_super._dark_orbs.append({"x": 500.0, "dir": -1.0, "hit": False})

hp_before = env_super.boss.hp

# Cast Ultra Plasma Shot
env_super.step(Action.USE_SUPER)
assert env_super.player.state == PlayerState.SUPER
assert env_super.player.iframes > 0, "Player should be invulnerable during super"

# Step through 0.4s cast
for _ in range(10):
    env_super.step(Action.IDLE)
    if env_super.player.state == PlayerState.IDLE:
        break

# Damage check: 80 dmg
assert env_super.boss.hp == hp_before - 80.0, f"Expected 80 dmg, got {hp_before - env_super.boss.hp}"
# Stagger check: boss stunned for 1.0s
assert env_super.boss.state == BossState.STUN, f"Boss should be stunned for 1.0s, got {env_super.boss.state.name}"
# Projectile clearing check: all projectiles cleared!
assert len(env_super._light_bolts) == 0, "Light bolts should be cleared by laser"
assert len(env_super._dark_waves) == 0, "Dark waves should be cleared by laser"
assert len(env_super._dark_orbs) == 0, "Dark orbs should be cleared by laser"
assert env_super.player.super_stacks == 0, "Super stacks should be consumed"

print(f"  Ultra Plasma Shot dealt 80 dmg, stunned boss, and cleared all projectiles (OK)\n")

print("All assertions passed (OK)")
