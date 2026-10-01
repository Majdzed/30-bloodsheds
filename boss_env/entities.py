"""Game entity dataclasses: Player and Boss.

All timer fields are in **frames** (integer counts) so they can be decremented
by 1 every simulation frame without floating-point drift.  Helper properties
expose the values in normalised [0,1] form for the observation builder.

Conventions
-----------
- Positions are in game units (u). Arena spans [0, ARENA_W].
- Timer = 0 means the action is ready / not active.
- Boss ability cooldowns are stored in a dict keyed by ability id (1..7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict

from . import constants as C


# ---------------------------------------------------------------------------
# State enumerations (match SPEC.md § 6 one-hot lists exactly)
# ---------------------------------------------------------------------------

class PlayerState(IntEnum):
    IDLE   = 0
    ATTACK = 1
    ROLL   = 2
    PARRY  = 3
    DRINK  = 4
    STUN   = 5
    SUPER  = 6


class BossState(IntEnum):
    IDLE   = 0
    WINDUP = 1
    ATTACK = 2
    HEAL   = 3
    STUN   = 4


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    """All mutable player state for one fight episode."""

    # --- spatial ---
    x: float = 150.0           # starting position near the left wall
    y: float = 0.0             # 0 = ground
    vy: float = 0.0            # vertical velocity (u/s, positive = up)

    # --- vitals ---
    hearts: int = C.BASE_HEARTS
    max_hearts: int = C.BASE_HEARTS
    stamina: float = C.STAMINA_MAX
    stamina_regen_delay: int = 0  # frames until stamina regen resumes

    # --- resources ---
    flasks: int = 1
    ammo: int = C.REVOLVER_MAG           # irrelevant for Katana but harmless
    reload_timer: int = 0                # frames remaining in active reload
    reload_idle_timer: int = 0           # frames until auto-reload triggers
    super_stacks: int = 0                # 0..3

    # --- state machine ---
    state: PlayerState = PlayerState.IDLE

    # --- action timers (frames remaining, 0 = ready) ---
    action_timer: int = 0        # generic: frames left in current action
    roll_cd: int = 0             # cooldown frames after a roll ends
    parry_cd: int = 0            # cooldown frames after a parry ends
    parry_active: int = 0        # frames the parry window is still open
    iframes: int = 0             # invulnerability frames (roll i-frames)

    # --- airborne ---
    airborne: bool = False

    # ---- derived / helpers ------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.hearts > 0

    @property
    def on_ground(self) -> bool:
        return not self.airborne

    @property
    def actionable(self) -> bool:
        """True when the player can accept a new action."""
        return self.state in (PlayerState.IDLE,)

    # Normalised values for the observation vector
    @property
    def hp_norm(self) -> float:
        return self.hearts / 5.0          # max hearts ever is 5 (SPEC §3)

    @property
    def stamina_norm(self) -> float:
        return self.stamina / C.STAMINA_MAX

    @property
    def roll_cd_norm(self) -> float:
        return self.roll_cd / C.sec(C.ROLL_COOLDOWN_S)

    @property
    def parry_cd_norm(self) -> float:
        return self.parry_cd / C.sec(C.PARRY_COOLDOWN_S)

    @property
    def reload_progress_norm(self) -> float:
        """1.0 = reload just started; 0.0 = full mag / not reloading."""
        total = C.sec(C.REVOLVER_RELOAD_S)
        return self.reload_timer / total if self.reload_timer > 0 else 0.0

    @property
    def super_norm(self) -> float:
        return self.super_stacks / C.SUPER_STACKS_REQUIRED

    @property
    def flask_norm(self) -> float:
        return self.flasks / 3.0          # max flasks ever is 3


# ---------------------------------------------------------------------------
# Boss
# ---------------------------------------------------------------------------

@dataclass
class Boss:
    """All mutable boss state for one fight episode."""

    # --- spatial ---
    x: float = 650.0           # starting position near the right wall

    # --- vitals ---
    hp: float = 150.0
    max_hp: float = 150.0

    # --- state machine ---
    state: BossState = BossState.IDLE
    current_ability_id: int = 0   # 0 = none, 1..7 matching ABILITIES

    # --- generic action timer ---
    action_timer: int = 0         # frames remaining in windup / active phase

    # --- ability cooldowns (frames remaining) ---
    ability_cd: Dict[int, int] = field(default_factory=lambda: {i: 0 for i in range(1, 8)})

    # --- blink ---
    blink_cd: int = 0             # frames remaining on the blink cooldown

    # --- siphon heal ---
    orbs_remaining: int = 0       # 0 when not healing
    orb_consume_timer: int = 0    # frames until the next orb is consumed

    # --- global recovery ---
    recovery_timer: int = 0       # 0.3 s global recovery after any action

    # ---- derived / helpers ------------------------------------------------

    @property
    def alive(self) -> bool:
        return self.hp > 0

    @property
    def hp_norm(self) -> float:
        return self.hp / self.max_hp

    @property
    def is_healing(self) -> bool:
        return self.state == BossState.HEAL

    @property
    def windup_timer_norm(self) -> float:
        """Fraction of the current windup that has elapsed, else 0."""
        # Spec doesn't give a fixed max, so we normalise to the longest windup (1.2 s)
        max_frames = C.sec(1.2)
        return self.action_timer / max_frames

    @property
    def blink_cd_norm(self) -> float:
        return self.blink_cd / C.sec(C.BLINK_COOLDOWN_S)
