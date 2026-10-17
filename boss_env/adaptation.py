"""Boss adaptation layer: Markov-style player model.

Only active when BossFightEnv is constructed with ``adaptive_boss=True`` (set
by play.py).  When the flag is off the env behaves identically to before this
file was added – same RNG consumption, same seed-reproducible trajectories.

Design (from SPEC.md §9 – Adaptive Boss)
-----------------------------------------
Context  = (boss_ability_id: int 1-7,  distance_bucket: str "close"|"mid"|"far")
Response = what the player did during that ability's windup + active window,
           collapsed to one of 7 discrete labels:
               roll, jump, parry, attack, move_toward, move_away, idle

A first-order count table counts[context][response] accumulates observations.
After SAMPLES_REF samples the boss weights its candidate list by:

    final_p = (1 - alpha) * uniform  +  alpha * learned
    alpha   = min(ALPHA_MAX, total_samples / SAMPLES_REF)

Counter matrix (from SPEC §3 and §5): which responses each ability exploits.
    jumps a lot      -> Lightning Strike (1), Light Bolt (4), Frenzy (7), Dark Orb (6)
    rolls a lot      -> Dark Orb (6)
    parries a lot    -> Dark Spikes (2)  [cannot be parried]
    stands still     -> Dark Spikes (2)
    stays far away   -> Dark Wave (3), Light Bolt (4), blink+Lightning Strike (1)

Learning schedule (constants below):
    LEARN_UNTIL_FIGHT : model freezes (stops accepting new observations) after this fight.
    APPLY_FROM_FIGHT  : weighted selection begins from this fight onwards.
                        Fights 1-3 only have Lightning Strike, so nothing to choose.

Persistence: the PlayerModel is created once per game session in play.py and
passed to every BossFightEnv via set_player_model().  It survives retries and
fight advances.  It resets when a new gauntlet starts (play.py calls
model.reset()).
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Tuneable constants (mirrored in SPEC.md §9)
# ---------------------------------------------------------------------------

ALPHA_MAX        = 0.6   # maximum boss-learns fraction; keeps it beatable
SAMPLES_REF      = 50    # observations needed to reach ALPHA_MAX
LEARN_UNTIL_FIGHT = 10   # stop accepting new observations after this fight index
APPLY_FROM_FIGHT  = 4    # start using weights from this fight (fights 1-3: only 1 candidate)

# ---------------------------------------------------------------------------
# Response labels (what the player did during a boss ability window)
# ---------------------------------------------------------------------------

RESPONSES: List[str] = [
    "roll",
    "jump",
    "parry",
    "attack",
    "move_toward",
    "move_away",
    "idle",
]

# ---------------------------------------------------------------------------
# Distance buckets
# ---------------------------------------------------------------------------

def distance_bucket(dist: float) -> str:
    """Map absolute player-boss distance to a bucket label."""
    if dist < 100.0:
        return "close"
    elif dist <= 300.0:
        return "mid"
    else:
        return "far"


# ---------------------------------------------------------------------------
# Counter matrix: ability_id -> set of responses it counters / exploits
# Derived from SPEC.md §3 & §5 ability table.
# ---------------------------------------------------------------------------
# Higher weight = ability is more effective against players showing that response.

_COUNTER_MATRIX: Dict[int, Dict[str, float]] = {
    # 1 – Lightning Strike: bad against roll (i-frames), bad against parry;
    #     effective against jump (jump does NOT avoid it), move_away (blink closes gap),
    #     and idle targets.
    1: {"roll": 0.1, "jump": 1.8, "parry": 0.2, "attack": 0.6,
        "move_toward": 1.2, "move_away": 1.4, "idle": 1.6},

    # 2 – Dark Spikes: CANNOT be parried; cannot be jumped (ground/slow);
    #     great against parry-heavy and stand-still players.
    2: {"roll": 0.5, "jump": 0.3, "parry": 2.0, "attack": 1.0,
        "move_toward": 1.5, "move_away": 0.3, "idle": 1.8},

    # 3 – Dark Wave: ground projectile, jump-clearable; good vs stay-far players.
    3: {"roll": 0.4, "jump": 0.3, "parry": 0.8, "attack": 0.5,
        "move_toward": 0.4, "move_away": 1.8, "idle": 1.2},

    # 4 – Light Bolt: high-speed projectile, NOT avoided by jump; parryable.
    4: {"roll": 0.6, "jump": 1.6, "parry": 0.4, "attack": 0.6,
        "move_toward": 0.8, "move_away": 1.6, "idle": 1.2},

    # 5 – Siphon Heal: not a damage ability; selection is handled separately by SPEC §5 rule 1.
    5: {"roll": 1.0, "jump": 1.0, "parry": 1.0, "attack": 0.8,
        "move_toward": 1.0, "move_away": 1.0, "idle": 1.0},

    # 6 – Ultimate Dark Orb: cannot be dodged, cannot be jumped; best vs roll-heavy players.
    6: {"roll": 2.0, "jump": 1.8, "parry": 0.3, "attack": 0.8,
        "move_toward": 1.0, "move_away": 1.0, "idle": 1.2},

    # 7 – Dual-Blade Frenzy: 3-hit dash, jump does NOT avoid it; roll i-frames do.
    7: {"roll": 0.2, "jump": 1.8, "parry": 0.4, "attack": 1.0,
        "move_toward": 1.4, "move_away": 0.6, "idle": 1.6},
}


# ---------------------------------------------------------------------------
# PlayerModel
# ---------------------------------------------------------------------------

class PlayerModel:
    """First-order Markov player behaviour model.

    Thread-safety: NOT thread-safe; designed for single-threaded play.py use.
    """

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset: start a new gauntlet.  Call from play.py when user
        returns to the main menu and starts a fresh run."""
        # counts[context_key][response] = int
        # context_key = (ability_id: int, distance_bucket: str)
        self._counts: Dict[Tuple[int, str], Dict[str, int]] = defaultdict(
            lambda: {r: 0 for r in RESPONSES}
        )
        self._total_samples: int = 0
        self._current_fight: int = 1

    def set_fight(self, fight: int) -> None:
        """Called by env.set_fight / play.py when advancing to a new fight."""
        self._current_fight = fight

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def observe(self, ability_id: int, dist: float, response: str) -> None:
        """Record one player-response observation.

        Parameters
        ----------
        ability_id : int
            Boss ability being used (1-7).
        dist : float
            Absolute player-boss distance at the start of the ability window.
        response : str
            One of RESPONSES; the dominant player behaviour during the window.
        """
        # Only learn during the designated learning window
        if self._current_fight > LEARN_UNTIL_FIGHT:
            return
        if response not in RESPONSES:
            return
        if ability_id not in range(1, 8):
            return

        ctx = (ability_id, distance_bucket(dist))
        self._counts[ctx][response] += 1
        self._total_samples += 1

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def tendencies(self) -> Dict[str, float]:
        """Overall player response rates across all contexts (0..1, sum ≈ 1).

        Returns a dict with one entry per RESPONSE label.
        If no data has been collected yet, returns uniform rates.
        """
        totals: Dict[str, float] = {r: 0.0 for r in RESPONSES}
        grand_total = 0
        for ctx_counts in self._counts.values():
            for r, n in ctx_counts.items():
                totals[r] += n
                grand_total += n

        if grand_total == 0:
            n = len(RESPONSES)
            return {r: 1.0 / n for r in RESPONSES}

        return {r: totals[r] / grand_total for r in RESPONSES}

    def weights_for_candidates(
        self, candidates: List[int], fight: int
    ) -> Optional[List[float]]:
        """Return a probability distribution over ``candidates`` (ability ids).

        Returns ``None`` if adaptation is not yet in effect (fight < APPLY_FROM_FIGHT,
        no data, or alpha effectively 0), so the caller falls back to uniform sampling.

        Parameters
        ----------
        candidates : List[int]
            Valid ability ids for the current frame (from the boss decision logic).
        fight : int
            Current fight index; used to enforce APPLY_FROM_FIGHT gate.
        """
        if fight < APPLY_FROM_FIGHT:
            return None
        if len(candidates) <= 1:
            # Nothing to choose from; uniform is trivial
            return None

        alpha = min(ALPHA_MAX, self._total_samples / SAMPLES_REF)
        if alpha < 1e-6:
            return None

        tend = self.tendencies()

        # Compute raw learned scores: for each candidate ability,
        # dot-product its counter weights with the current tendency.
        raw: List[float] = []
        for aid in candidates:
            matrix_row = _COUNTER_MATRIX.get(aid, {r: 1.0 for r in RESPONSES})
            score = sum(matrix_row.get(r, 1.0) * tend.get(r, 0.0) for r in RESPONSES)
            raw.append(max(score, 1e-8))

        # Normalize to probability vector
        raw_sum = sum(raw)
        learned_p = [v / raw_sum for v in raw]

        n = len(candidates)
        uniform_p = [1.0 / n] * n

        # Blend
        final_p = [
            (1.0 - alpha) * u + alpha * lp
            for u, lp in zip(uniform_p, learned_p)
        ]

        # Normalise again (should already sum to 1, but float arithmetic)
        total = sum(final_p)
        return [v / total for v in final_p]

    @property
    def alpha(self) -> float:
        """Current blend factor (0..ALPHA_MAX)."""
        return min(ALPHA_MAX, self._total_samples / SAMPLES_REF)

    @property
    def total_samples(self) -> int:
        return self._total_samples

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Serialise to JSON string for save/load."""
        serialisable = {
            "total_samples": self._total_samples,
            "current_fight": self._current_fight,
            # Convert tuple keys to strings for JSON
            "counts": {
                f"{k[0]}|{k[1]}": dict(v)
                for k, v in self._counts.items()
            },
        }
        return json.dumps(serialisable, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "PlayerModel":
        """Deserialise from a JSON string produced by to_json()."""
        data = json.loads(s)
        model = cls()
        model._total_samples = int(data.get("total_samples", 0))
        model._current_fight = int(data.get("current_fight", 1))
        for raw_key, response_counts in data.get("counts", {}).items():
            parts = raw_key.split("|", 1)
            if len(parts) != 2:
                continue
            aid_str, bucket = parts
            try:
                aid = int(aid_str)
            except ValueError:
                continue
            ctx = (aid, bucket)
            merged = {r: 0 for r in RESPONSES}
            for r, n in response_counts.items():
                if r in merged:
                    merged[r] = int(n)
            model._counts[ctx] = merged
        return model

    # ------------------------------------------------------------------
    # Human-readable summary (for HUD display)
    # ------------------------------------------------------------------

    def top_tendencies_text(self, n: int = 3) -> List[str]:
        """Return up to n human-readable tendency strings for HUD display.

        E.g.  ["You roll 62% of slashes", "You stand still 38% of bolts"]
        """
        if self._total_samples == 0:
            return ["No data yet – keep fighting!"]

        tend = self.tendencies()
        sorted_r = sorted(tend.items(), key=lambda kv: kv[1], reverse=True)

        labels = {
            "roll": "rolls",
            "jump": "jumps",
            "parry": "parries",
            "attack": "attacks",
            "move_toward": "moves toward boss",
            "move_away": "moves away",
            "idle": "stands still",
        }

        lines: List[str] = []
        for r, rate in sorted_r[:n]:
            if rate > 0.01:
                lines.append(f"You {labels[r]} {rate * 100:.0f}% of the time")
        return lines if lines else ["Not enough data yet"]
