"""Headless Gymnasium environment for the boss gauntlet.

NO drawing code here. Rendering lives in renderer.py and only READS env state.

Status: movement + boss Lightning Strike + player heart damage + episode termination.
        Katana attacks, reward shaping, roll/parry/jump come next.
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
        self._frame: int = 0          # total frames simulated (never resets)
        self._fight_frame: int = 0    # frames elapsed in the current fight
        self._terminated: bool = False
        # Frames in the boss's current windup (for obs normalisation).
        # Set when the boss enters BossState.WINDUP, cleared on IDLE.
        self._boss_windup_total: int = 1   # safe default, never 0

    # ---- gauntlet control -------------------------------------------------
    def set_fight(self, fight: int) -> None:
        """Select the fight (1..30). Takes effect on the next reset()."""
        self.fight = int(np.clip(fight, 1, C.N_FIGHTS))

    # ---- Gymnasium API ----------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._fight_frame = 0
        self._terminated  = False
        self._boss_windup_total = 1

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
        reward = 0.0
        terminated = False
        truncated  = False

        for _ in range(self.frame_skip):
            self._tick(action)
            if self._terminated:
                terminated = True
                break
            if self._fight_frame >= C.FIGHT_TIME_LIMIT_FRAMES:
                truncated = True
                break

        # TODO: accumulate reward inside _tick and sum here
        reward += C.REWARD["step"]

        return self._get_obs(), reward, terminated, truncated, {}

    # ---- Core simulation: one frame ---------------------------------------
    def _tick(self, action: int) -> None:
        """Advance the game by exactly one frame (1/60 s)."""
        self._frame       += 1
        self._fight_frame += 1
        p = self.player
        b = self.boss

        dt = 1.0 / C.FPS   # seconds per frame

        # ---- Player movement (only when not locked into an action) ----
        if p.state == PlayerState.IDLE:
            if action == Action.MOVE_LEFT:
                p.x -= C.PLAYER_SPEED * dt
            elif action == Action.MOVE_RIGHT:
                p.x += C.PLAYER_SPEED * dt
        p.x = float(np.clip(p.x, 0.0, C.ARENA_W))

        dist = abs(b.x - p.x)

        # ================================================================
        # Boss state machine
        # ================================================================
        # --- countdown any active timer ---
        if b.action_timer > 0:
            b.action_timer -= 1

        # --- decrement global recovery ---
        if b.recovery_timer > 0:
            b.recovery_timer -= 1

        # --- decrement ability cooldowns ---
        for aid in b.ability_cd:
            if b.ability_cd[aid] > 0:
                b.ability_cd[aid] -= 1

        if b.state == BossState.WINDUP:
            # Windup finished → transition to ATTACK
            if b.action_timer == 0:
                ab = C.ABILITY_FRAMES[b.current_ability_id]
                b.state        = BossState.ATTACK
                b.action_timer = ab["active"]

        elif b.state == BossState.ATTACK:
            # --- Lightning Strike hit detection ---
            if b.current_ability_id == 1:
                # Damage fires on the FIRST frame of the active window
                # Use a one-shot flag on Boss to avoid multi-hitting.
                if not b.attack_hit_landed:
                    if dist <= 120.0:          # Lightning Strike range: 120 u
                        p.hearts -= 1
                        if p.hearts <= 0:
                            self._terminated = True
                    b.attack_hit_landed = True

            # Active phase finished → recovery
            if b.action_timer == 0:
                b.state              = BossState.IDLE
                b.current_ability_id = 0
                b.attack_hit_landed  = False
                b.recovery_timer     = C.sec(C.BOSS_GLOBAL_RECOVERY_S)

        elif b.state == BossState.IDLE:
            # Walk toward player when out of melee range
            if dist > C.MELEE_TRIGGER_DIST:
                direction = 1.0 if p.x > b.x else -1.0
                b.x += direction * C.BOSS_SPEED * dt
                b.x = float(np.clip(b.x, 0.0, C.ARENA_W))

            # --- Ability decision (simplified: Lightning Strike only for now) ---
            if b.recovery_timer == 0:
                # Lightning Strike: player within 100 u, ability off cooldown
                if dist <= C.MELEE_TRIGGER_DIST and b.ability_cd[1] == 0:
                    ab = C.ABILITY_FRAMES[1]
                    b.state              = BossState.WINDUP
                    b.current_ability_id = 1
                    b.action_timer       = ab["windup"]
                    b.attack_hit_landed  = False
                    b.ability_cd[1]      = ab["cooldown"]
                    self._boss_windup_total = ab["windup"]

    # ---- Action masks -----------------------------------------------------
    def action_masks(self) -> np.ndarray:
        """Boolean mask over Action. Used by MaskablePPO (sb3-contrib)."""
        mask = np.ones(N_ACTIONS, dtype=bool)
        if self.player is None:
            # Before first reset – block everything except IDLE
            mask[:] = False
            mask[Action.IDLE] = True
            return mask

        p = self.player

        # Movement walls
        mask[Action.MOVE_LEFT]  = p.x > 0.0
        mask[Action.MOVE_RIGHT] = p.x < C.ARENA_W

        # Resources (full logic comes with combat; stubs are spec-correct)
        mask[Action.USE_SUPER] = p.super_stacks >= C.SUPER_STACKS_REQUIRED
        mask[Action.USE_FLASK] = p.flasks > 0 and p.hearts < p.max_hearts

        # Everything combat-gated is not yet implemented – leave unmasked for now
        # TODO: JUMP, LIGHT_ATTACK, HEAVY_ATTACK, DODGE_ROLL, PARRY when not actionable

        return mask


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
            # boss_windup_timer: fraction of windup remaining (1→0 as boss approaches attack)
            if b.state == BossState.WINDUP and self._boss_windup_total > 0:
                obs[_scalar_idx("boss_windup_timer")] = b.action_timer / self._boss_windup_total
            else:
                obs[_scalar_idx("boss_windup_timer")] = 0.0
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
