"""Rule-based Katana and Revolver agents for Boss Gauntlet.

Per SPEC.md Section 8:
"Baseline: rule-based agent, one version per weapon, evaluated on the same fights and seeds."

Plugs into the standard Stable-Baselines3 evaluation and benchmarking interface:
    action, _ = agent.predict(obs, mask=mask)

Capabilities:
- Katana: Spacing (~90-120u), melee punish, parry counters, Blood Katana super.
- Revolver: Fullscreen kiting (250-500u), ammo conservation, piercing heavy against orbs,
            safe reloading, Ultra Plasma Shot laser super.
- Universal Defense:
    1. Ultimate Dark Orb (6): Must parry (cannot be dodged).
    2. Dark Wave (3): Must jump over ground projectile.
    3. Dark Spikes (2): Roll or step out of the warning radius.
    4. Light Bolt (4): Parry (builds super) or roll.
    5. Melee / Frenzy (1 & 7): Parry near end of windup or dodge roll.
    6. Siphon Heal (5): Piercing heavy or melee to destroy orbs and interrupt channel.
    7. Flask Healing: Safe drink when low HP.
"""

from typing import Optional, Tuple
import numpy as np

from boss_env import constants as C
from boss_env.env import (
    Action,
    OBS_SCALARS,
    _PS_START,
    _BS_START,
    _BA_START,
    PLAYER_STATES,
    BOSS_STATES,
    BOSS_ATTACKS,
)


class RuleBasedAgent:
    def __init__(self, weapon: str = "katana"):
        self.weapon = weapon.lower()
        self.idx_x_rel = OBS_SCALARS.index("x_rel")
        self.idx_player_x = OBS_SCALARS.index("player_x")
        self.idx_player_y = OBS_SCALARS.index("player_y")
        self.idx_player_vy = OBS_SCALARS.index("player_vy")
        self.idx_hp = OBS_SCALARS.index("player_hp")
        self.idx_stamina = OBS_SCALARS.index("player_stamina")
        self.idx_ammo = OBS_SCALARS.index("ammo_level")
        self.idx_reload = OBS_SCALARS.index("reload_progress")
        self.idx_super = OBS_SCALARS.index("super_meter")
        self.idx_flasks = OBS_SCALARS.index("flask_count")
        self.idx_boss_hp = OBS_SCALARS.index("boss_hp")
        self.idx_windup = OBS_SCALARS.index("boss_windup_timer")
        self.idx_is_healing = OBS_SCALARS.index("boss_is_healing")
        self.idx_orbs = OBS_SCALARS.index("heal_orbs_remaining")
        self.idx_spike_rel_x = OBS_SCALARS.index("spike_rel_x")
        self.idx_spike_timer = OBS_SCALARS.index("spike_timer")
        self.idx_proj_dark = OBS_SCALARS.index("proj_dark_rel_x")
        self.idx_proj_light = OBS_SCALARS.index("proj_light_rel_x")
        self.idx_orb_parry = OBS_SCALARS.index("orb_parry_window")

    def predict(
        self,
        obs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        action_masks: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[int, None]:
        """Predict next action given observation vector and action mask."""
        if mask is None:
            mask = action_masks
        if mask is None:
            mask = np.ones(len(Action), dtype=bool)

        # -------------------------------------------------------------
        # 1. Parse Observation Features
        # -------------------------------------------------------------
        x_rel = float(obs[self.idx_x_rel])            # (boss.x - player.x) / 800
        dist = abs(x_rel) * C.ARENA_W                 # absolute distance in units
        boss_dir = 1.0 if x_rel >= 0 else -1.0        # +1 if boss is to the right

        player_x = float(obs[self.idx_player_x]) * C.ARENA_W
        player_y = float(obs[self.idx_player_y]) * 100.0
        player_hp_norm = float(obs[self.idx_hp])
        player_stamina = float(obs[self.idx_stamina]) * 100.0
        ammo_level = int(round(float(obs[self.idx_ammo]) * C.REVOLVER_MAG))
        reload_progress = float(obs[self.idx_reload])
        super_stacks = int(round(float(obs[self.idx_super]) * C.SUPER_STACKS_REQUIRED))

        boss_windup = float(obs[self.idx_windup])     # fraction remaining
        boss_is_healing = float(obs[self.idx_is_healing]) > 0.5
        orbs_remaining = int(round(float(obs[self.idx_orbs]) * 4.0))

        spike_rel_x = float(obs[self.idx_spike_rel_x])
        spike_timer = float(obs[self.idx_spike_timer])
        proj_dark_rel_x = float(obs[self.idx_proj_dark])
        proj_light_rel_x = float(obs[self.idx_proj_light])
        orb_parry_window = float(obs[self.idx_orb_parry])

        # State one-hots
        ps_idx = int(np.argmax(obs[_PS_START:_BS_START]))
        player_state_name = PLAYER_STATES[ps_idx]
        bs_idx = int(np.argmax(obs[_BS_START:_BA_START]))
        boss_state_name = BOSS_STATES[bs_idx]
        ba_idx = int(np.argmax(obs[_BA_START:_BA_START + len(BOSS_ATTACKS)]))
        boss_attack_name = BOSS_ATTACKS[ba_idx]

        # Directional navigation
        move_toward = Action.MOVE_RIGHT if boss_dir > 0 else Action.MOVE_LEFT
        move_away   = Action.MOVE_LEFT  if boss_dir > 0 else Action.MOVE_RIGHT

        # =============================================================
        # PRIORITY 1: CRITICAL THREAT DEFENSE (Universal)
        # =============================================================

        # 1A. Ultimate Dark Orb (Ability 6) - CANNOT DODGE, MUST PARRY!
        if orb_parry_window > 0.5:
            if mask[Action.PARRY]:
                return int(Action.PARRY), None

        # 1B. Dark Wave (Ability 3) - Ground projectile, MUST JUMP OVER IT
        if abs(proj_dark_rel_x) > 0.001:
            wave_dist = abs(proj_dark_rel_x) * C.ARENA_W
            # Dark Wave moves at 280 u/s (approx 19 u per 4-frame step)
            if wave_dist <= 110.0 and player_y < 10.0:
                if mask[Action.JUMP]:
                    return int(Action.JUMP), None

        # 1C. Light Bolt (Ability 4) - High projectile (600 u/s), Parry or Roll
        if abs(proj_light_rel_x) > 0.001:
            bolt_dist = abs(proj_light_rel_x) * C.ARENA_W
            if bolt_dist <= 90.0:
                if mask[Action.PARRY]:
                    return int(Action.PARRY), None
                elif mask[Action.DODGE_ROLL]:
                    return int(Action.DODGE_ROLL), None

        # 1D. Dark Spikes (Ability 2) - 140u diameter circle on player's position
        if spike_timer > 0.001:
            spike_dist = abs(spike_rel_x) * C.ARENA_W
            if spike_dist <= 75.0 and spike_timer <= 0.45:
                # Inside spike radius and about to erupt!
                if mask[Action.DODGE_ROLL]:
                    return int(Action.DODGE_ROLL), None
                # If roll not ready, walk away from spike center
                spike_dir = 1.0 if spike_rel_x >= 0 else -1.0
                escape_action = Action.MOVE_LEFT if spike_dir > 0 else Action.MOVE_RIGHT
                if mask[escape_action]:
                    return int(escape_action), None

        # 1E. Melee attacks (Lightning Strike ab 1, Dual-Blade Frenzy ab 7)
        if boss_state_name == "windup" or boss_windup > 0.0:
            if dist <= 140.0:
                # Reaction threshold: parry when windup is nearly complete (<= 0.25)
                # Windup is 30 frames (0.5s); <= 0.25 means <= 7 frames remaining.
                if boss_windup <= 0.25:
                    if mask[Action.PARRY]:
                        return int(Action.PARRY), None
                    elif mask[Action.DODGE_ROLL]:
                        return int(Action.DODGE_ROLL), None
                elif boss_windup <= 0.50 and not mask[Action.PARRY]:
                    # If parry not available, roll earlier
                    if mask[Action.DODGE_ROLL]:
                        return int(Action.DODGE_ROLL), None
                # Still early in windup: create space if possible
                if mask[move_away]:
                    return int(move_away), None

        # 1F. Dual-Blade Frenzy active flurry (ab 7) or Lightning Strike attack frame
        if boss_state_name == "attack" and dist <= 130.0:
            if mask[Action.PARRY]:
                return int(Action.PARRY), None
            elif mask[Action.DODGE_ROLL]:
                return int(Action.DODGE_ROLL), None

        # =============================================================
        # PRIORITY 2: SIPHON HEAL (Ability 5) INTERRUPT
        # =============================================================
        if boss_is_healing or boss_state_name == "heal":
            if self.weapon == "revolver":
                # Piercing heavy attack destroys all orbs at full range!
                if mask[Action.HEAVY_ATTACK]:
                    return int(Action.HEAVY_ATTACK), None
                elif mask[Action.LIGHT_ATTACK]:
                    return int(Action.LIGHT_ATTACK), None
            else:
                # Katana: run up to boss and attack orbs
                if dist <= 120.0:
                    if mask[Action.HEAVY_ATTACK]:
                        return int(Action.HEAVY_ATTACK), None
                    elif mask[Action.LIGHT_ATTACK]:
                        return int(Action.LIGHT_ATTACK), None
                elif mask[move_toward]:
                    return int(move_toward), None

        # =============================================================
        # PRIORITY 3: SUPER ATTACK EXECUTION
        # =============================================================
        if mask[Action.USE_SUPER]:
            if self.weapon == "revolver":
                # Ultra Plasma Shot: fullscreen laser, fires from any range!
                return int(Action.USE_SUPER), None
            else:
                # Blood Katana: 200u reach
                if dist <= 190.0:
                    return int(Action.USE_SUPER), None
                elif mask[move_toward]:
                    return int(move_toward), None

        # =============================================================
        # PRIORITY 4: FLASK HEALING
        # =============================================================
        if player_hp_norm <= 0.40 and mask[Action.USE_FLASK]:
            # Drink when safe (boss far or stunned, and no active projectile)
            safe_dist = 200.0 if self.weapon == "katana" else 250.0
            if (dist > safe_dist or boss_state_name == "stun") and abs(proj_light_rel_x) < 0.001 and abs(proj_dark_rel_x) < 0.001:
                return int(Action.USE_FLASK), None

        # =============================================================
        # PRIORITY 5: WEAPON-SPECIFIC COMBAT & SPACING
        # =============================================================
        if self.weapon == "revolver":
            return self._predict_revolver(
                obs, mask, dist, player_x, ammo_level, reload_progress,
                boss_state_name, move_toward, move_away
            )
        else:
            return self._predict_katana(
                obs, mask, dist, boss_state_name, move_toward, move_away
            )

    def _predict_katana(
        self, obs, mask, dist, boss_state_name, move_toward, move_away
    ) -> Tuple[int, None]:
        """Katana melee combat logic."""
        # When boss is stunned: close in and punish hard!
        if boss_state_name == "stun":
            if dist <= 140.0 and mask[Action.HEAVY_ATTACK]:
                return int(Action.HEAVY_ATTACK), None
            elif dist <= 115.0 and mask[Action.LIGHT_ATTACK]:
                return int(Action.LIGHT_ATTACK), None
            elif dist > 110.0 and mask[move_toward]:
                return int(move_toward), None

        # When boss is idle or in attack recovery
        if boss_state_name in ("idle", "attack"):
            # Attack if in range
            if dist <= 110.0:
                if mask[Action.HEAVY_ATTACK]:
                    return int(Action.HEAVY_ATTACK), None
                elif mask[Action.LIGHT_ATTACK]:
                    return int(Action.LIGHT_ATTACK), None

            # Spacing: maintain melee range ~95-115u
            if dist > 115.0:
                if mask[move_toward]:
                    return int(move_toward), None
            elif dist < 85.0 and not mask[Action.LIGHT_ATTACK] and not mask[Action.HEAVY_ATTACK]:
                if mask[move_away]:
                    return int(move_away), None

        # Fallback
        return self._fallback(mask)

    def _predict_revolver(
        self, obs, mask, dist, player_x, ammo_level, reload_progress,
        boss_state_name, move_toward, move_away
    ) -> Tuple[int, None]:
        """Revolver ranged combat and kiting logic."""
        # If out of ammo and reloading: prioritize safe kiting
        if ammo_level == 0:
            if dist < 220.0 and mask[move_away]:
                return int(move_away), None
            if mask[Action.IDLE]:
                return int(Action.IDLE), None

        # Boss is stunned: heavy attack punish!
        if boss_state_name == "stun":
            if mask[Action.HEAVY_ATTACK] and ammo_level >= 2:
                return int(Action.HEAVY_ATTACK), None
            elif mask[Action.LIGHT_ATTACK]:
                return int(Action.LIGHT_ATTACK), None

        # When boss is too close (< 140u), step back
        if dist < 140.0 and mask[move_away]:
            return int(move_away), None

        # Fire attacks when at safe range
        # Heavy attack if boss is far (> 300u) and ammo >= 2
        if dist > 300.0 and mask[Action.HEAVY_ATTACK] and ammo_level >= 2:
            return int(Action.HEAVY_ATTACK), None
        # Light attack is fast (9 frames windup) and reliable
        if mask[Action.LIGHT_ATTACK] and ammo_level >= 1:
            return int(Action.LIGHT_ATTACK), None

        # Spacing: avoid cornering against walls
        if player_x < 80.0 and mask[Action.MOVE_RIGHT]:
            return int(Action.MOVE_RIGHT), None
        elif player_x > 720.0 and mask[Action.MOVE_LEFT]:
            return int(Action.MOVE_LEFT), None

        # If very far (> 450u), close distance slightly
        if dist > 450.0 and mask[move_toward]:
            return int(move_toward), None

        return self._fallback(mask)

    def _fallback(self, mask: np.ndarray) -> Tuple[int, None]:
        if mask[Action.IDLE]:
            return int(Action.IDLE), None
        legal = np.where(mask)[0]
        if len(legal) > 0:
            return int(legal[0]), None
        return int(Action.IDLE), None


# Convenient function export
_default_katana_agent = RuleBasedAgent(weapon="katana")
_default_revolver_agent = RuleBasedAgent(weapon="revolver")


def predict(obs: np.ndarray, mask: Optional[np.ndarray] = None, weapon: str = "katana", **kwargs) -> Tuple[int, None]:
    agent = _default_revolver_agent if weapon == "revolver" else _default_katana_agent
    return agent.predict(obs, mask=mask, **kwargs)
