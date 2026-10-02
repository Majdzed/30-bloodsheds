"""Rule-based Katana agent for Boss Gauntlet.

Plugs into the same evaluation and benchmarking interface as the RL agent:
    action, _ = agent.predict(obs, mask=mask)

Strategy:
1. Parry / Roll when boss is in windup and within melee reach.
2. Parry / Roll when Light Bolt projectile is incoming and close.
3. Use Blood Katana Super (3 stacks) when within reach (200u).
4. Punish boss with Heavy/Light attacks during boss Stun or Recovery.
5. Maintain optimal melee spacing (~90-120u) and engage when safe.
6. Heal with Flask when safe (boss far or stunned) and hearts low.
"""

from typing import Optional, Tuple, Union
import numpy as np

from boss_env import constants as C
from boss_env.env import Action, OBS_SCALARS, _PS_START, _BS_START, _BA_START, PLAYER_STATES, BOSS_STATES, BOSS_ATTACKS


class RuleBasedAgent:
    def __init__(self, weapon: str = "katana"):
        self.weapon = weapon
        self.idx_x_rel = OBS_SCALARS.index("x_rel")
        self.idx_hp = OBS_SCALARS.index("player_hp")
        self.idx_super = OBS_SCALARS.index("super_meter")
        self.idx_boss_hp = OBS_SCALARS.index("boss_hp")
        self.idx_windup = OBS_SCALARS.index("boss_windup_timer")
        self.idx_proj_light = OBS_SCALARS.index("proj_light_rel_x")

    def predict(
        self,
        obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, None]:
        """Predict the next action given observation and action mask.

        Returns (action_id, None) matching the Stable-Baselines3 interface.
        """
        # Accept mask via either parameter name
        if mask is None:
            mask = action_masks
        if mask is None:
            mask = np.ones(len(Action), dtype=bool)

        # 1. Parse observation features
        x_rel = float(obs[self.idx_x_rel])            # (boss.x - player.x) / 800
        dist = abs(x_rel) * C.ARENA_W                 # absolute distance in units
        boss_dir = 1.0 if x_rel >= 0 else -1.0        # +1 if boss is to the right

        player_hp_norm = float(obs[self.idx_hp])
        super_norm = float(obs[self.idx_super])
        boss_windup = float(obs[self.idx_windup])     # fraction of windup remaining
        proj_light_rel_x = float(obs[self.idx_proj_light])

        # Boss state from one-hot block
        bs_one_hot = obs[_BS_START : _BS_START + len(BOSS_STATES)]
        bs_idx = int(np.argmax(bs_one_hot)) if len(bs_one_hot) > 0 else 0
        boss_state_name = BOSS_STATES[bs_idx]         # 'idle', 'windup', 'attack', 'heal', 'stun'

        # Boss attack ability from one-hot block
        ba_one_hot = obs[_BA_START : _BA_START + len(BOSS_ATTACKS)]
        ba_idx = int(np.argmax(ba_one_hot)) if len(ba_one_hot) > 0 else 0
        boss_attack_name = BOSS_ATTACKS[ba_idx]       # 'none', 'Lightning Strike', ...

        # Movement actions
        move_toward = Action.MOVE_RIGHT if boss_dir > 0 else Action.MOVE_LEFT
        move_away   = Action.MOVE_LEFT  if boss_dir > 0 else Action.MOVE_RIGHT

        # =====================================================================
        # Priority 1: Defend against incoming Light Bolt projectile
        # =====================================================================
        if abs(proj_light_rel_x) > 0.001:
            bolt_dist = abs(proj_light_rel_x) * C.ARENA_W
            # Speed is 600 u/s (10 u/frame, 40 u/step)
            if bolt_dist <= 85.0:
                if mask[Action.PARRY]:
                    return int(Action.PARRY), None
                elif mask[Action.DODGE_ROLL]:
                    return int(Action.DODGE_ROLL), None

        # =====================================================================
        # Priority 2: Defend against Boss Windup (Lightning Strike)
        # =====================================================================
        if boss_state_name == "windup" or boss_windup > 0.0:
            if dist <= 130.0:
                # Parry if inside reaction window (windup timer near expiration or close)
                if boss_windup <= 0.55:
                    if mask[Action.PARRY]:
                        return int(Action.PARRY), None
                    elif mask[Action.DODGE_ROLL]:
                        return int(Action.DODGE_ROLL), None
                # If still early in windup, create distance or prepare
                if mask[move_away]:
                    return int(move_away), None

        # =====================================================================
        # Priority 3: Unleash Blood Katana Super when in range (200u reach)
        # =====================================================================
        if mask[Action.USE_SUPER] and dist <= 180.0:
            return int(Action.USE_SUPER), None

        # =====================================================================
        # Priority 4: Heal with Flask if low HP and safe
        # =====================================================================
        if player_hp_norm <= 0.4 and mask[Action.USE_FLASK]:
            if dist > 200.0 or boss_state_name == "stun":
                return int(Action.USE_FLASK), None

        # =====================================================================
        # Priority 5: Punish Boss when Stunned
        # =====================================================================
        if boss_state_name == "stun":
            if dist <= 140.0 and mask[Action.HEAVY_ATTACK]:
                return int(Action.HEAVY_ATTACK), None
            elif dist <= 115.0 and mask[Action.LIGHT_ATTACK]:
                return int(Action.LIGHT_ATTACK), None
            elif dist > 110.0 and mask[move_toward]:
                return int(move_toward), None

        # =====================================================================
        # Priority 6: Attack & Spacing when Boss is Idle / Recovering
        # =====================================================================
        if boss_state_name in ("idle", "attack"):
            # If in attack range, strike!
            if dist <= 110.0:
                if mask[Action.HEAVY_ATTACK]:
                    return int(Action.HEAVY_ATTACK), None
                elif mask[Action.LIGHT_ATTACK]:
                    return int(Action.LIGHT_ATTACK), None

            # Spacing: close in if out of melee reach
            if dist > 115.0:
                if mask[move_toward]:
                    return int(move_toward), None
            elif dist < 85.0 and not mask[Action.LIGHT_ATTACK] and not mask[Action.HEAVY_ATTACK]:
                # Back up slightly if right up in boss's face and on attack cooldown
                if mask[move_away]:
                    return int(move_away), None

        # =====================================================================
        # Fallback: Pick any valid legal action
        # =====================================================================
        if mask[Action.IDLE]:
            return int(Action.IDLE), None
        legal = np.where(mask)[0]
        if len(legal) > 0:
            return int(legal[0]), None
        return int(Action.IDLE), None


# Convenient function export
_default_agent = RuleBasedAgent(weapon="katana")


def predict(obs: np.ndarray, mask: Optional[np.ndarray] = None, **kwargs) -> Tuple[int, None]:
    return _default_agent.predict(obs, mask=mask, **kwargs)
