"""Quick sanity check for entities + obs. Run with: python check_obs.py"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from boss_env import constants as C
from boss_env.env import BossFightEnv, OBS_SCALARS, _PS_START, _BS_START, _BA_START, PLAYER_STATES, BOSS_STATES, BOSS_ATTACKS

FIGHT_SAMPLES = [1, 8, 15, 26, 30]

print(f"{'FIGHT':>5}  {'boss_max_hp':>11}  {'obs[fight_idx]':>14}  {'obs[boss_hp]':>12}  "
      f"{'obs[player_hp]':>14}  {'obs[x_rel]':>10}  player_state  boss_state  boss_attack")
print("-" * 100)

for fight in FIGHT_SAMPLES:
    env = BossFightEnv(weapon="katana", fight=fight)
    obs, _ = env.reset()

    fi        = obs[OBS_SCALARS.index("fight_index")]
    boss_hp   = obs[OBS_SCALARS.index("boss_hp")]
    player_hp = obs[OBS_SCALARS.index("player_hp")]
    x_rel     = obs[OBS_SCALARS.index("x_rel")]

    ps_hot = obs[_PS_START : _PS_START + len(PLAYER_STATES)]
    bs_hot = obs[_BS_START : _BS_START + len(BOSS_STATES)]
    ba_hot = obs[_BA_START : _BA_START + len(BOSS_ATTACKS)]

    ps = PLAYER_STATES[ps_hot.argmax()]
    bs = BOSS_STATES  [bs_hot.argmax()]
    ba = BOSS_ATTACKS [ba_hot.argmax()]

    # Verify boss_hp == 1.0 at start
    expected_hp = C.boss_max_hp(fight)
    assert abs(boss_hp - 1.0) < 1e-5, f"fight {fight}: boss_hp norm should be 1.0, got {boss_hp}"
    assert abs(fi - fight / C.N_FIGHTS) < 1e-5, f"fight {fight}: fight_index mismatch"

    print(f"{fight:>5}  {expected_hp:>11}  {fi:>14.4f}  {boss_hp:>12.4f}  "
          f"{player_hp:>14.4f}  {x_rel:>10.4f}  {ps:^13} {bs:^11} {ba}")

print()
print("Revolver check (fight 26):")
env2 = BossFightEnv(weapon="revolver", fight=26)
obs2, _ = env2.reset()
wt = obs2[OBS_SCALARS.index("weapon_type")]
al = obs2[OBS_SCALARS.index("ammo_level")]
ul_ids = [i+1 for i, b in enumerate(obs2[OBS_SCALARS.index("unlocked_1"):OBS_SCALARS.index("unlocked_1")+7]) if b > 0.5]
print(f"  weapon_type={wt:.1f} (1.0=revolver)  ammo_level={al:.4f}  unlocked abilities: {ul_ids}")

print()
print("All assertions passed (OK)")
