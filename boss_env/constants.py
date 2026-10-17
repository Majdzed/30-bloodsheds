"""Numeric game parameters. Mirrors SPEC.md. Change both together."""

from dataclasses import dataclass

# ---- Timing / world -------------------------------------------------------
FPS = 60
ARENA_W = 800.0
N_FIGHTS = 30
FRAME_SKIP = 4                 # agent acts every 4 frames (15 decisions/s)
FIGHT_TIME_LIMIT_S = 90.0


def sec(seconds: float) -> int:
    """Seconds -> frames at FPS."""
    return int(round(seconds * FPS))


FIGHT_TIME_LIMIT_FRAMES = sec(FIGHT_TIME_LIMIT_S)

# ---- Player ---------------------------------------------------------------
BASE_HEARTS = 4
MAX_HEARTS = 5                 # after the fight-5 health upgrade
PLAYER_SPEED = 220.0           # u/s (proposed)

STAMINA_MAX = 100.0
STAMINA_REGEN_PER_S = 25.0
STAMINA_REGEN_DELAY_S = 0.5
ROLL_COST = 30.0
ROLL_DURATION_S = 0.35         # (proposed)
ROLL_DISTANCE = 120.0          # (proposed)
ROLL_IFRAMES_S = 0.2           # (proposed)
ROLL_COOLDOWN_S = 0.6          # (proposed)

JUMP_AIRTIME_S = 0.6
JUMP_APEX = 100.0
# Discrete 60 FPS kinematics for exact apex = 100 u and exact airtime = 0.6 s (36 frames):
# N = 18 frames to apex. g = JUMP_APEX / (N*(N-1)/2 * dt²), vy0 = N * g * dt
_N_HALF = round((JUMP_AIRTIME_S / 2.0) * FPS)
JUMP_GRAVITY    = JUMP_APEX / (_N_HALF * (_N_HALF - 1) / 2.0 * (1.0 / FPS)**2)  # ≈ 2352.94 u/s²
JUMP_VY_INITIAL = _N_HALF * JUMP_GRAVITY * (1.0 / FPS)                            # ≈ 705.88 u/s

PARRY_WINDOW_FRAMES = 12       # 0.2 s. Must exceed FRAME_SKIP (see SPEC.md).
PARRY_COOLDOWN_S = 1.0
SUPER_STACKS_REQUIRED = 3

FLASK_HEAL_HEARTS = 2
FLASK_DRINK_S = 0.6            # (proposed)


def flasks_for_fight(fight: int, upgrade: str = "health") -> int:
    """Flasks available at the start of a fight (1-indexed). upgrade: 'health' | 'flask'."""
    n = 1 if fight <= 3 else 2
    if fight >= 6 and upgrade == "flask":
        n += 1
    return n


def max_hearts_for_fight(fight: int, upgrade: str = "health") -> int:
    if fight >= 6 and upgrade == "health":
        return BASE_HEARTS + 1
    return BASE_HEARTS


# ---- Weapons --------------------------------------------------------------
@dataclass(frozen=True)
class Weapon:
    name: str
    light_dmg: float
    light_windup_s: float
    light_range: float
    light_recovery_s: float
    heavy_dmg: float
    heavy_windup_s: float
    heavy_range: float
    heavy_recovery_s: float
    super_name: str
    super_dmg: float
    super_duration_s: float
    uses_ammo: bool = False


KATANA = Weapon(
    name="katana",
    light_dmg=15, light_windup_s=0.20, light_range=120, light_recovery_s=0.40,
    heavy_dmg=35, heavy_windup_s=0.50, heavy_range=150, heavy_recovery_s=0.40,
    super_name="Blood Katana", super_dmg=90, super_duration_s=0.8,
)

REVOLVER = Weapon(
    name="revolver",
    light_dmg=6, light_windup_s=0.15, light_range=ARENA_W, light_recovery_s=0.15,
    heavy_dmg=18, heavy_windup_s=0.60, heavy_range=ARENA_W, heavy_recovery_s=0.25,
    super_name="Ultra Plasma Shot", super_dmg=80, super_duration_s=0.4,
    uses_ammo=True,
)

WEAPONS = {"katana": KATANA, "revolver": REVOLVER}

REVOLVER_MAG = 6
REVOLVER_HEAVY_COST = 2
REVOLVER_RELOAD_S = 1.2        # (proposed)
REVOLVER_IDLE_RELOAD_DELAY_S = 0.5

# ---- Boss -----------------------------------------------------------------
BOSS_SPEED = 120.0
BOSS_GLOBAL_RECOVERY_S = 0.3   # (proposed)

MELEE_TRIGGER_DIST = 100.0
BLINK_MIN_DIST = 150.0
BLINK_COOLDOWN_S = 8.0

HEAL_ORB_COUNT = 4
HEAL_ORB_INTERVAL_S = 0.75
HEAL_PER_ORB = 15
HEAL_TRIGGER_HP_FRACTION = 0.60  # (proposed)


def boss_max_hp(fight: int) -> int:
    """150 at fight 1, 350 at fight 15, 600 at fight 30, linear in between."""
    if fight <= 15:
        return round(150 + (fight - 1) * (350 - 150) / 14)
    return round(350 + (fight - 15) * (600 - 350) / 15)


# ---- Reward ---------------------------------------------------------------
REWARD = {
    "damage_dealt": 1.0,       # x (damage / boss max HP)
    "heart_lost": -0.25,
    "parry": 0.3,
    "orb_destroyed": 0.1,
    "heal_interrupted": 0.3,
    "win": 1.0,
    "loss": -1.0,              # also used for timeout
    "step": -0.001,
}
# ---- Ability timing (frames). Mirrors SPEC.md §5 ability table. ----------
# Keyed by ability id (1..7). All values in frames at FPS=60.
ABILITY_FRAMES: dict = {
    1: {"windup": sec(0.50), "active": sec(0.15), "cooldown": sec(2.0)},   # Lightning Strike
    2: {"windup": sec(0.80), "active": sec(0.25), "cooldown": sec(5.0)},   # Dark Spikes
    3: {"windup": sec(0.60), "active": 0,          "cooldown": sec(4.0)},   # Dark Wave (projectile)
    4: {"windup": sec(0.30), "active": 0,          "cooldown": sec(2.0)},   # Light Bolt (projectile)
    5: {"windup": sec(0.75), "active": sec(3.0),   "cooldown": sec(14.0)},  # Siphon Heal
    6: {"windup": sec(1.20), "active": sec(2.0),   "cooldown": sec(18.0)},  # Ultimate Dark Orb
    7: {"windup": sec(0.25), "active": sec(0.30),  "cooldown": sec(3.5)},   # Dual-Blade Frenzy
}

# ---- Projectile and Hazard constants ---------------------------------------
LIGHT_BOLT_SPEED        = 600.0   # u/s (SPEC §5 ability table)
LIGHT_BOLT_HIT_RADIUS   = 15.0    # u – tolerance for hit detection on a point projectile

DARK_SPIKES_RADIUS      = 70.0    # 140 u circle at player x when cast
DARK_WAVE_SPEED         = 280.0   # u/s ground projectile
DARK_WAVE_HIT_RADIUS    = 20.0    # u

FRENZY_DASH_SPEED       = 500.0   # 150 u over 0.30 s dash
FRENZY_STRIKE_INTERVAL  = 6       # 3 strikes x 0.10 s (6 frames each at 60 FPS)

DARK_ORB_SPEED          = 400.0   # 400 u/s (SPEC §5 ability table)
DARK_ORB_HIT_RADIUS     = 25.0    # u
DARK_ORB_PARRY_DIST     = 80.0    # u (distance where orb_parry_window obs becomes 1.0)

# ---- Boss Dodge constants (human mode only, SPEC §10) ----------------------
DODGE_UNLOCK_FIGHT = 10
DODGE_IFRAMES_S = 0.25         # 15 frames of invulnerability
DODGE_DISTANCE = 120.0         # u moved away from player
DODGE_RECOVERY_S = 0.30        # 18 frames vulnerable after dodge
DODGE_MAX_PER_10S = 2          # rate limit: max 2 dodges in a rolling 10s window


def boss_react_time_s(fight: int) -> float:
    """Reaction threshold in seconds: 0.20s at fight 10 shrinking to 0.10s at fight 30."""
    f = max(10, min(30, fight))
    return 0.20 - (f - 10) * (0.20 - 0.10) / 20.0


def boss_dodge_prob(fight: int) -> float:
    """Dodge trigger probability: 0.5 at fight 10 rising to 0.8 at fight 30."""
    f = max(10, min(30, fight))
    return 0.50 + (f - 10) * (0.80 - 0.50) / 20.0


def boss_dodge_cooldown_s(fight: int) -> float:
    """Dodge cooldown in seconds: 5.0s at fight 10 shrinking to 3.5s at fight 30."""
    f = max(10, min(30, fight))
    return 5.0 - (f - 10) * (5.0 - 3.5) / 20.0


