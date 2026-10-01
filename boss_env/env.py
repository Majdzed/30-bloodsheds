"""Headless Gymnasium environment for the boss gauntlet.

NO drawing code here. Rendering lives in renderer.py and only READS env state.

Status: entities initialised; observation filled for spatial + one-hot features.
        step() still placeholder – movement comes next.
"""

from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import constants as C
from .abilities import ABILITIES, unlocked_mask
from .entities import Boss, BossState, Player, PlayerState


class Action(IntEnum):
    IDLE         = 0
    MOVE_LEFT    = 1
    MOVE_RIGHT   = 2
    JUMP         = 3
    LIGHT_ATTACK = 4
    HEAVY_ATTACK = 5
    DODGE_ROLL   = 6
    PARRY        = 7
    USE_FLASK    = 8
    USE_SUPER    = 9


N_ACTIONS = len(Action)

# ---- Observation layout (52 features). Keep in sync with SPEC.md ----------
OBS_SCALARS = [
    # player (14)
    "x_rel", "player_x", "player_y", "player_vy", "player_hp", "player_stamina",
    "weapon_type", "ammo_level", "reload_progress", "super_meter", "flask_count",
    "roll_cd", "parry_cd", "parry_active",
    # boss (5)
    "boss_hp", "boss_windup_timer", "boss_is_healing", "heal_orbs_remaining",
    "boss_blink_cd",
    # hazards (5)
    "spike_rel_x", "spike_timer", "proj_dark_rel_x", "proj_light_rel_x",
    "orb_parry_window",
    # context, agent-only, never drawn in the HUD (8)
    "fight_index", *[f"unlocked_{a.id}" for a in ABILITIES],
]
PLAYER_STATES = [s.name.lower() for s in PlayerState]   # ["idle","attack","roll","parry","drink","stun","super"]
BOSS_STATES   = [s.name.lower() for s in BossState]     # ["idle","windup","attack","heal","stun"]
BOSS_ATTACKS  = ["none"] + [a.name for a in ABILITIES]  # 8 entries

OBS_SIZE = len(OBS_SCALARS) + len(PLAYER_STATES) + len(BOSS_STATES) + len(BOSS_ATTACKS)
assert OBS_SIZE == 52, OBS_SIZE

# Pre-compute slice offsets for the one-hot blocks
_SCALAR_END   = len(OBS_SCALARS)                           # 32
_PS_START     = _SCALAR_END                                # 32
_PS_END       = _PS_START + len(PLAYER_STATES)             # 39
_BS_START     = _PS_END                                    # 39
_BS_END       = _BS_START + len(BOSS_STATES)               # 44
_BA_START     = _BS_END                                    # 44
_BA_END       = _BA_START + len(BOSS_ATTACKS)              # 52


def _scalar_idx(name: str) -> int:
    return OBS_SCALARS.index(name)


class BossFightEnv(gym.Env):
    """One episode = one fight. The gauntlet is driven from outside via set_fight()."""

    metadata = {"render_modes": ["human"], "render_fps": C.FPS}

    def __init__(self, weapon="katana", fight=1, upgrade="health",
                 frame_skip=C.FRAME_SKIP, render_mode=None):
        super().__init__()
        assert weapon in C.WEAPONS and upgrade in ("health", "flask")
        assert frame_skip < C.PARRY_WINDOW_FRAMES, "parry window must exceed frame skip"
        self.weapon      = C.WEAPONS[weapon]
        self.fight       = fight
        self.upgrade     = upgrade
        self.frame_skip  = frame_skip
        self.render_mode = render_mode

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

        # entities are None until reset()
        self.player: Player | None = None
        self.boss:   Boss   | None = None
        self._t = 0

    # ---- gauntlet control -------------------------------------------------
    def set_fight(self, fight: int) -> None:
        """Select the fight (1..30). Takes effect on the next reset()."""
        self.fight = int(np.clip(fight, 1, C.N_FIGHTS))

    # ---- Gymnasium API ----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0

        max_hp = C.boss_max_hp(self.fight)
        max_hearts = C.max_hearts_for_fight(self.fight, self.upgrade)
        flasks     = C.flasks_for_fight(self.fight, self.upgrade)

        self.player = Player(
            x          = 150.0,
            y          = 0.0,
            vy         = 0.0,
            hearts     = max_hearts,
            max_hearts = max_hearts,
            stamina    = C.STAMINA_MAX,
            flasks     = flasks,
            ammo       = C.REVOLVER_MAG,
        )

        self.boss = Boss(
            x      = 650.0,
            hp     = float(max_hp),
            max_hp = float(max_hp),
        )

        return self._get_obs(), {}

    def step(self, action):
        # TODO: apply action for self.frame_skip frames, run boss AI, collect reward.
        self._t += 1
        terminated = self._t >= 100          # PLACEHOLDER
        return self._get_obs(), 0.0, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        """Boolean mask over Action. Used by MaskablePPO (sb3-contrib)."""
        mask = np.ones(N_ACTIONS, dtype=bool)
        # TODO: real conditions from SPEC.md §6; placeholders for now
        if self.player is not None:
            mask[Action.USE_SUPER] = self.player.super_stacks >= C.SUPER_STACKS_REQUIRED
            mask[Action.USE_FLASK] = self.player.flasks > 0 and \
                                     self.player.hearts < self.player.max_hearts
        else:
            mask[Action.USE_SUPER] = False
            mask[Action.USE_FLASK] = False
        return mask

    # ---- Observation builder ---------------------------------------------
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        p   = self.player
        b   = self.boss

        # ---- scalars ----
        if p is not None and b is not None:
            # player group (14)
            obs[_scalar_idx("x_rel")]          = (b.x - p.x) / C.ARENA_W
            obs[_scalar_idx("player_x")]       = p.x / C.ARENA_W
            obs[_scalar_idx("player_y")]       = p.y / C.JUMP_APEX        # 0..1 during a jump
            obs[_scalar_idx("player_vy")]      = np.clip(p.vy / 600.0, -1.0, 1.0)
            obs[_scalar_idx("player_hp")]      = p.hp_norm
            obs[_scalar_idx("player_stamina")] = p.stamina_norm
            obs[_scalar_idx("weapon_type")]    = 0.0 if self.weapon.name == "katana" else 1.0
            obs[_scalar_idx("ammo_level")]     = p.ammo / C.REVOLVER_MAG
            obs[_scalar_idx("reload_progress")]= p.reload_progress_norm
            obs[_scalar_idx("super_meter")]    = p.super_norm
            obs[_scalar_idx("flask_count")]    = p.flask_norm
            obs[_scalar_idx("roll_cd")]        = p.roll_cd_norm
            obs[_scalar_idx("parry_cd")]       = p.parry_cd_norm
            obs[_scalar_idx("parry_active")]   = 1.0 if p.parry_active > 0 else 0.0

            # boss group (5)
            obs[_scalar_idx("boss_hp")]             = b.hp_norm
            obs[_scalar_idx("boss_windup_timer")]   = b.windup_timer_norm
            obs[_scalar_idx("boss_is_healing")]     = 1.0 if b.is_healing else 0.0
            obs[_scalar_idx("heal_orbs_remaining")] = b.orbs_remaining / C.HEAL_ORB_COUNT
            obs[_scalar_idx("boss_blink_cd")]       = b.blink_cd_norm

            # hazards (5) – zeroed until hazard simulation is added
            # spike_rel_x, spike_timer, proj_dark_rel_x, proj_light_rel_x, orb_parry_window

            # context (8)
            obs[_scalar_idx("fight_index")] = self.fight / C.N_FIGHTS
            for a, bit in zip(ABILITIES, unlocked_mask(self.fight)):
                obs[_scalar_idx(f"unlocked_{a.id}")] = float(bit)

            # ---- player state one-hot (7) ----
            obs[_PS_START + int(p.state)] = 1.0

            # ---- boss state one-hot (5) ----
            obs[_BS_START + int(b.state)] = 1.0

            # ---- boss attack one-hot (8: none=0, abilities 1..7) ----
            obs[_BA_START + b.current_ability_id] = 1.0

        else:
            # before first reset – return all zeros (safe; obs space allows 0)
            pass

        return obs
