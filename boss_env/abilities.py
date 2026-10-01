"""Boss ability table and unlock schedule. Mirrors SPEC.md section 5.

Adding an ability = one new row here plus its behavior in env.py.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Ability:
    id: int                        # 1..7, 0 is reserved for "none"
    name: str
    windup_s: float
    active_s: float                # 0 for projectiles / channels handled separately
    damage_hearts: int
    cooldown_s: float
    unlock_fight: int
    parryable: bool
    jump_clears: bool              # ground-level hit: a jump avoids it
    dodgeable: bool = True         # False only for the Ultimate Dark Orb
    range_u: Optional[float] = None
    speed_u_s: Optional[float] = None


ABILITIES: List[Ability] = [
    Ability(1, "Lightning Strike", windup_s=0.50, active_s=0.15, damage_hearts=1,
            cooldown_s=2.0, unlock_fight=1, parryable=True, jump_clears=False,
            range_u=120.0),
    Ability(2, "Dark Spikes", windup_s=0.80, active_s=0.25, damage_hearts=1,
            cooldown_s=5.0, unlock_fight=8, parryable=False, jump_clears=True,
            range_u=140.0),
    Ability(3, "Dark Wave", windup_s=0.60, active_s=0.0, damage_hearts=1,
            cooldown_s=4.0, unlock_fight=12, parryable=True, jump_clears=True,
            speed_u_s=280.0),
    Ability(4, "Light Bolt", windup_s=0.30, active_s=0.0, damage_hearts=1,
            cooldown_s=2.0, unlock_fight=4, parryable=True, jump_clears=False,
            speed_u_s=600.0),
    Ability(5, "Siphon Heal", windup_s=0.75, active_s=3.0, damage_hearts=0,
            cooldown_s=14.0, unlock_fight=21, parryable=False, jump_clears=False),
    Ability(6, "Ultimate Dark Orb", windup_s=1.20, active_s=2.0, damage_hearts=3,
            cooldown_s=18.0, unlock_fight=26, parryable=True, jump_clears=False,
            dodgeable=False, speed_u_s=400.0),
    Ability(7, "Dual-Blade Frenzy", windup_s=0.25, active_s=0.30, damage_hearts=1,
            cooldown_s=3.5, unlock_fight=16, parryable=True, jump_clears=False,
            range_u=150.0),
]

ABILITY_BY_ID = {a.id: a for a in ABILITIES}


def unlocked_abilities(fight: int) -> List[Ability]:
    """Abilities available to the boss in a given fight (1-indexed)."""
    return [a for a in ABILITIES if a.unlock_fight <= fight]


def unlocked_mask(fight: int) -> List[int]:
    """7-bit mask in ability-id order, used as an observation feature."""
    return [1 if a.unlock_fight <= fight else 0 for a in ABILITIES]
