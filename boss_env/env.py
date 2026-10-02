"""Headless Gymnasium environment for the boss gauntlet.

NO drawing code here. Rendering lives in renderer.py and only READS env state.

Status: Katana combat complete with abilities 1 (Lightning Strike) and 4 (Light Bolt),
        Jump, Roll, Parry, Super (Blood Katana), and rewards.
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
        # Reward accumulated inside _tick, drained each step()
        self._tick_reward: float = 0.0
        # Per-episode stats (for info dict)
        self._hearts_lost: int   = 0
        self._damage_dealt: float = 0.0
        # Active Light Bolt projectiles: each is {"x": float, "dir": +1 or -1}
        self._light_bolts: list = []

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
        self._tick_reward   = 0.0
        self._hearts_lost   = 0
        self._damage_dealt  = 0.0
        self._light_bolts   = []

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
        self._tick_reward = 0.0   # reset accumulator at the start of each decision
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

        reward = self._tick_reward + C.REWARD["step"]

        # --- Terminal bonuses / penalties ---
        if terminated:
            if self.boss.hp <= 0:
                reward += C.REWARD["win"]
            else:
                reward += C.REWARD["loss"]   # player died
        elif truncated:
            reward += C.REWARD["loss"]       # timeout counts as a loss

        info = {}
        if terminated or truncated:
            info = {
                "win":          self.boss.hp <= 0,
                "hearts_lost":  self._hearts_lost,
                "damage_dealt": self._damage_dealt,
                "fight_frames": self._fight_frame,
                "fight":        self.fight,
            }

        return self._get_obs(), reward, terminated, truncated, info

    # ---- Core simulation: one frame ---------------------------------------
    def _tick(self, action: int) -> None:
        """Advance the game by exactly one frame (1/60 s)."""
        self._frame       += 1
        self._fight_frame += 1
        p = self.player
        b = self.boss

        dt = 1.0 / C.FPS   # seconds per frame

        # ================================================================
        # Player state machine
        # ================================================================

        # --- Tick down active timers ---
        if p.action_timer > 0:
            p.action_timer -= 1
        if p.recovery_timer > 0:
            p.recovery_timer -= 1
        if p.roll_cd > 0:
            p.roll_cd -= 1
        if p.parry_cd > 0:
            p.parry_cd -= 1
        if p.iframes > 0:
            p.iframes -= 1

        # --- Stamina regen ---
        if p.stamina_regen_delay > 0:
            p.stamina_regen_delay -= 1
        elif p.stamina < C.STAMINA_MAX:
            p.stamina = min(C.STAMINA_MAX, p.stamina + C.STAMINA_REGEN_PER_S * dt)

        if p.state == PlayerState.ATTACK:
            if p.action_timer == 0:
                # --- Damage resolution: end of windup ---
                dist = abs(b.x - p.x)
                if p.pending_attack_id == 1:        # light attack
                    if dist <= self.weapon.light_range:
                        dmg = min(self.weapon.light_dmg, b.hp)
                        b.hp -= dmg
                        self._tick_reward  += C.REWARD["damage_dealt"] * (dmg / b.max_hp)
                        self._damage_dealt += dmg
                        if b.hp <= 0:
                            b.hp = 0
                            self._terminated = True
                    p.recovery_timer = C.sec(self.weapon.light_recovery_s)
                elif p.pending_attack_id == 2:      # heavy attack
                    if dist <= self.weapon.heavy_range:
                        dmg = min(self.weapon.heavy_dmg, b.hp)
                        b.hp -= dmg
                        self._tick_reward  += C.REWARD["damage_dealt"] * (dmg / b.max_hp)
                        self._damage_dealt += dmg
                        if b.hp <= 0:
                            b.hp = 0
                            self._terminated = True
                    p.recovery_timer = C.sec(self.weapon.heavy_recovery_s)
                p.pending_attack_id = 0
                p.state = PlayerState.IDLE    # recovery phase: player is IDLE but locked by recovery_timer

        elif p.state == PlayerState.ROLL:
            # Move the player continuously during the roll
            roll_speed = C.ROLL_DISTANCE / C.ROLL_DURATION_S   # u/s during roll
            p.x += p.roll_dir * roll_speed * dt
            p.x  = float(np.clip(p.x, 0.0, C.ARENA_W))
            # Roll ends when action_timer reaches 0
            if p.action_timer == 0:
                p.state   = PlayerState.IDLE
                p.roll_cd = C.sec(C.ROLL_COOLDOWN_S)

        elif p.state == PlayerState.PARRY:
            # Parry window is active for PARRY_WINDOW_FRAMES frames (tracked by action_timer)
            if p.parry_active > 0:
                p.parry_active -= 1
            if p.action_timer == 0:
                p.state    = PlayerState.IDLE
                p.parry_cd = C.sec(C.PARRY_COOLDOWN_S)

        elif p.state == PlayerState.SUPER:
            # Blood Katana: player invulnerable, damage at end of lock
            if p.action_timer == 0:
                dist = abs(b.x - p.x)
                if dist <= 200.0:              # Blood Katana reach: 200 u
                    dmg = min(self.weapon.super_dmg, b.hp)
                    b.hp -= dmg
                    self._tick_reward  += C.REWARD["damage_dealt"] * (dmg / b.max_hp)
                    self._damage_dealt += dmg
                    if b.hp <= 0:
                        b.hp = 0
                        self._terminated = True
                p.super_stacks = 0
                p.state        = PlayerState.IDLE

        elif p.state == PlayerState.DRINK:
            # Drinking flask takes 0.6s (36 frames). At end, heal 2 hearts.
            if p.action_timer == 0:
                p.flasks -= 1
                p.hearts = min(p.max_hearts, p.hearts + C.FLASK_HEAL_HEARTS)
                p.state  = PlayerState.IDLE

        elif p.state == PlayerState.IDLE:
            # Accept new action input when recovery has also cleared
            if p.recovery_timer == 0:
                if action == Action.LIGHT_ATTACK and not p.airborne:
                    p.state             = PlayerState.ATTACK
                    p.pending_attack_id = 1
                    p.action_timer      = C.sec(self.weapon.light_windup_s)
                elif action == Action.HEAVY_ATTACK and not p.airborne:
                    p.state             = PlayerState.ATTACK
                    p.pending_attack_id = 2
                    p.action_timer      = C.sec(self.weapon.heavy_windup_s)
                elif action == Action.DODGE_ROLL and not p.airborne:
                    if p.stamina >= C.ROLL_COST and p.roll_cd == 0:
                        p.stamina            -= C.ROLL_COST
                        p.stamina_regen_delay = C.sec(C.STAMINA_REGEN_DELAY_S)
                        p.state               = PlayerState.ROLL
                        p.action_timer        = C.sec(C.ROLL_DURATION_S)
                        p.iframes             = C.sec(C.ROLL_IFRAMES_S)
                        # Roll away from the boss; if at wall, roll toward it
                        if b.x > p.x:
                            p.roll_dir = -1.0 if p.x > 0 else 1.0
                        else:
                            p.roll_dir =  1.0 if p.x < C.ARENA_W else -1.0
                elif action == Action.PARRY:
                    if p.parry_cd == 0:
                        p.state        = PlayerState.PARRY
                        p.action_timer = C.PARRY_WINDOW_FRAMES
                        p.parry_active = C.PARRY_WINDOW_FRAMES
                elif action == Action.USE_SUPER:
                    if p.super_stacks >= C.SUPER_STACKS_REQUIRED:
                        p.state        = PlayerState.SUPER
                        p.action_timer = C.sec(self.weapon.super_duration_s)
                        p.iframes      = p.action_timer   # invulnerable entire duration
                elif action == Action.USE_FLASK and not p.airborne:
                    if p.flasks > 0 and p.hearts < p.max_hearts:
                        p.state        = PlayerState.DRINK
                        p.action_timer = C.sec(C.FLASK_DRINK_S)
                elif action == Action.JUMP and not p.airborne:
                    p.vy       = C.JUMP_VY_INITIAL
                    p.airborne = True

        # ---- Jump physics (applied every tick, independent of state) ----
        if p.airborne:
            p.vy -= C.JUMP_GRAVITY * dt
            p.y  += p.vy * dt
            if p.y <= 0.0:             # landed
                p.y       = 0.0
                p.vy      = 0.0
                p.airborne = False

        # ---- Player horizontal movement (only when idle, on ground, not recovering) ----
        if p.state == PlayerState.IDLE and p.recovery_timer == 0 and not p.airborne:
            if action == Action.MOVE_LEFT:
                p.x -= C.PLAYER_SPEED * dt
            elif action == Action.MOVE_RIGHT:
                p.x += C.PLAYER_SPEED * dt
        p.x = float(np.clip(p.x, 0.0, C.ARENA_W))

        dist = abs(b.x - p.x)

        # ================================================================
        # Light Bolt projectile physics
        # ================================================================
        new_bolts = []
        for bolt in self._light_bolts:
            prev_x = bolt["x"]
            bolt["x"] += bolt["dir"] * C.LIGHT_BOLT_SPEED * dt
            # Swept and radius collision check
            crossed = (prev_x - p.x) * (bolt["x"] - p.x) <= 0
            close = abs(bolt["x"] - p.x) <= C.LIGHT_BOLT_HIT_RADIUS
            if (crossed or close) and not bolt.get("hit"):
                # Light Bolt is NOT avoided by jump (SPEC §3)
                if p.parry_active > 0:
                    # Parried: negate, +1 super, no stun (projectile ≠ melee)
                    p.super_stacks = min(p.super_stacks + 1, C.SUPER_STACKS_REQUIRED)
                    self._tick_reward += C.REWARD["parry"]
                elif p.iframes == 0:
                    p.hearts -= 1
                    self._tick_reward += C.REWARD["heart_lost"]
                    self._hearts_lost += 1
                    if p.hearts <= 0:
                        self._terminated = True
                bolt["hit"] = True           # consumed, don't carry it forward
            # Keep bolts that haven't left the arena and haven't hit
            elif 0.0 <= bolt["x"] <= C.ARENA_W and not bolt.get("hit"):
                new_bolts.append(bolt)
        self._light_bolts = new_bolts

        # ================================================================
        # Boss state machine

        # ================================================================
        # --- countdown any active timer ---
        if b.action_timer > 0:
            b.action_timer -= 1

        # --- decrement global recovery ---
        if b.recovery_timer > 0:
            b.recovery_timer -= 1

        # --- decrement blink cooldown ---
        if b.blink_cd > 0:
            b.blink_cd -= 1

        # --- decrement ability cooldowns ---
        for aid in b.ability_cd:
            if b.ability_cd[aid] > 0:
                b.ability_cd[aid] -= 1

        if b.state == BossState.WINDUP:
            # Teleport during slash windup if blink was triggered
            if b.blink_pending and b.action_timer <= (self._boss_windup_total // 2):
                target_x = p.x + (80.0 if b.x > p.x else -80.0)
                b.x = float(np.clip(target_x, 0.0, C.ARENA_W))
                b.blink_pending = False

            # Windup finished → transition to ATTACK (or spawn projectile)
            if b.action_timer == 0:
                ab = C.ABILITY_FRAMES[b.current_ability_id]
                if b.current_ability_id == 4:
                    # Light Bolt: projectile fired immediately, no active phase
                    direction = 1.0 if p.x > b.x else -1.0
                    self._light_bolts.append({"x": b.x, "dir": direction, "hit": False})
                    b.state              = BossState.IDLE
                    b.current_ability_id = 0
                    b.recovery_timer     = C.sec(C.BOSS_GLOBAL_RECOVERY_S)
                else:
                    b.state        = BossState.ATTACK
                    b.action_timer = ab["active"]

        elif b.state == BossState.ATTACK:
            # --- Lightning Strike hit detection ---
            if b.current_ability_id == 1:
                # Damage fires on the FIRST frame of the active window
                # Use a one-shot flag on Boss to avoid multi-hitting.
                if not b.attack_hit_landed:
                    if dist <= 120.0:
                        if p.parry_active > 0:
                            # --- Successful parry ---
                            # Hit is negated; +1 super stack; melee → stun boss 0.5 s
                            p.super_stacks = min(p.super_stacks + 1, C.SUPER_STACKS_REQUIRED)
                            self._tick_reward += C.REWARD["parry"]
                            # Lightning Strike is melee → stun
                            b.state        = BossState.STUN
                            b.action_timer = C.sec(0.5)
                        elif p.iframes == 0:   # i-frames block all non-Orb hits
                            p.hearts -= 1
                            self._tick_reward += C.REWARD["heart_lost"]
                            self._hearts_lost += 1
                            if p.hearts <= 0:
                                self._terminated = True
                    b.attack_hit_landed = True

            # Active phase finished → recovery
            if b.action_timer == 0:
                b.state              = BossState.IDLE
                b.current_ability_id = 0
                b.attack_hit_landed  = False
                b.recovery_timer     = C.sec(C.BOSS_GLOBAL_RECOVERY_S)

        elif b.state == BossState.STUN:
            # Boss stunned: wait out action_timer, then recover
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

            # --- Ability decision (abilities 1 and 4, spec-correct distance rules) ---
            if b.recovery_timer == 0:
                candidates = []
                # Lightning Strike: player within 100 u (or beyond 150 u with blink ready)
                if (dist <= C.MELEE_TRIGGER_DIST or (dist > C.BLINK_MIN_DIST and b.blink_cd == 0)) and b.ability_cd[1] == 0:
                    candidates.append(1)
                # Light Bolt: player beyond 150 u, unlocked (fight >= 4)
                if (dist > C.BLINK_MIN_DIST and b.ability_cd[4] == 0
                        and self.fight >= 4):
                    candidates.append(4)

                if candidates:
                    chosen = int(self.np_random.choice(candidates))
                    ab = C.ABILITY_FRAMES[chosen]
                    b.state              = BossState.WINDUP
                    b.current_ability_id = chosen
                    b.action_timer       = ab["windup"]
                    b.attack_hit_landed  = False
                    b.ability_cd[chosen] = ab["cooldown"]
                    self._boss_windup_total = ab["windup"]
                    if chosen == 1 and dist > C.BLINK_MIN_DIST and b.blink_cd == 0:
                        b.blink_pending = True
                        b.blink_cd = C.sec(C.BLINK_COOLDOWN_S)


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

        # "Not actionable" = attack windup, recovery, rolling, parrying, drinking, or in super (SPEC §6)
        not_actionable = (
            p.state in (PlayerState.ATTACK, PlayerState.ROLL,
                        PlayerState.PARRY, PlayerState.SUPER,
                        PlayerState.DRINK)
            or p.recovery_timer > 0
        )

        # Movement also blocked while not actionable
        if not_actionable:
            mask[Action.MOVE_LEFT]  = False
            mask[Action.MOVE_RIGHT] = False

        # Attacks: blocked if not actionable or airborne (no air attacks)
        mask[Action.LIGHT_ATTACK] = not not_actionable and not p.airborne
        mask[Action.HEAVY_ATTACK] = not not_actionable and not p.airborne

        # Dodge roll: needs stamina and cooldown clear; blocked if not actionable or airborne
        mask[Action.DODGE_ROLL] = (
            not not_actionable
            and not p.airborne
            and p.stamina >= C.ROLL_COST
            and p.roll_cd == 0
        )

        # Parry: needs cooldown clear; blocked while not actionable
        mask[Action.PARRY] = not not_actionable and p.parry_cd == 0

        # Jump: blocked if already airborne or not actionable
        mask[Action.JUMP] = not not_actionable and not p.airborne

        # Resources
        mask[Action.USE_SUPER] = (
            p.super_stacks >= C.SUPER_STACKS_REQUIRED
            and not not_actionable
            and not p.airborne
        )
        mask[Action.USE_FLASK] = (
            p.flasks > 0
            and p.hearts < p.max_hearts
            and not not_actionable
            and not p.airborne
        )

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

            # hazards (5)
            # spike_rel_x, spike_timer, proj_dark_rel_x, proj_light_rel_x, orb_parry_window
            if self._light_bolts:
                closest_bolt = min(self._light_bolts, key=lambda bolt: abs(bolt["x"] - p.x))
                obs[_scalar_idx("proj_light_rel_x")] = np.clip((closest_bolt["x"] - p.x) / C.ARENA_W, -1.0, 1.0)
            else:
                obs[_scalar_idx("proj_light_rel_x")] = 0.0

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
