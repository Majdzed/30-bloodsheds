"""Headless Gymnasium environment for the boss gauntlet.

NO drawing code here. Rendering lives in renderer.py and only READS env state.

Status: skeleton. reset/step are PLACEHOLDERS so the training pipeline can be
wired end to end. Replace them with the real simulation (see SPEC.md).
"""

from enum import IntEnum

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import constants as C
from .abilities import ABILITIES, unlocked_mask


class Action(IntEnum):
    IDLE = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    JUMP = 3
    LIGHT_ATTACK = 4
    HEAVY_ATTACK = 5
    DODGE_ROLL = 6
    PARRY = 7
    USE_FLASK = 8
    USE_SUPER = 9


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
PLAYER_STATES = ["idle", "attack", "roll", "parry", "drink", "stun", "super"]
BOSS_STATES = ["idle", "windup", "attack", "heal", "stun"]
BOSS_ATTACKS = ["none"] + [a.name for a in ABILITIES]

OBS_SIZE = len(OBS_SCALARS) + len(PLAYER_STATES) + len(BOSS_STATES) + len(BOSS_ATTACKS)
assert OBS_SIZE == 52, OBS_SIZE


class BossFightEnv(gym.Env):
    """One episode = one fight. The gauntlet is driven from outside via `fight`."""

    metadata = {"render_modes": ["human"], "render_fps": C.FPS}

    def __init__(self, weapon="katana", fight=1, upgrade="health",
                 frame_skip=C.FRAME_SKIP, render_mode=None):
        super().__init__()
        assert weapon in C.WEAPONS and upgrade in ("health", "flask")
        assert frame_skip < C.PARRY_WINDOW_FRAMES, "parry window must exceed frame skip"
        self.weapon = C.WEAPONS[weapon]
        self.fight = fight
        self.upgrade = upgrade
        self.frame_skip = frame_skip
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)

        self._t = 0

    # ---- gauntlet control -------------------------------------------------
    def set_fight(self, fight: int) -> None:
        """Select the fight (1..30). Takes effect on the next reset()."""
        self.fight = int(np.clip(fight, 1, C.N_FIGHTS))

    # ---- Gymnasium API ----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        # TODO: init player, boss (HP = C.boss_max_hp(self.fight)), flasks, timers
        return self._get_obs(), {}

    def step(self, action):
        # TODO: apply action for self.frame_skip frames, run boss AI, collect reward.
        self._t += 1
        terminated = self._t >= 100          # PLACEHOLDER
        return self._get_obs(), 0.0, terminated, False, {}

    def action_masks(self) -> np.ndarray:
        """Boolean mask over Action. Used by MaskablePPO (sb3-contrib)."""
        mask = np.ones(N_ACTIONS, dtype=bool)
        # TODO: real conditions, see SPEC.md section 6
        mask[Action.USE_SUPER] = False
        mask[Action.USE_FLASK] = False
        return mask

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        # TODO: fill from game state. Context features can already be set:
        obs[OBS_SCALARS.index("fight_index")] = self.fight / C.N_FIGHTS
        for a, bit in zip(ABILITIES, unlocked_mask(self.fight)):
            obs[OBS_SCALARS.index(f"unlocked_{a.id}")] = bit
        obs[OBS_SCALARS.index("weapon_type")] = 0.0 if self.weapon.name == "katana" else 1.0
        return obs
