"""Unit tests for the adaptive boss PlayerModel (boss_env/adaptation.py).

Run with:
    uv run python tests/test_adaptation.py

Tests:
  1. Counts update correctly and tendencies sum sensibly
  2. Weights sum to 1; alpha never exceeds ALPHA_MAX
  3. Model freezes after LEARN_UNTIL_FIGHT
  4. Regression: with adaptive_boss=False the same seed produces identical boss
     ability sequences as before the change
  5. Model JSON round-trips losslessly
"""

import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from boss_env.adaptation import (
    PlayerModel,
    ALPHA_MAX,
    SAMPLES_REF,
    LEARN_UNTIL_FIGHT,
    APPLY_FROM_FIGHT,
    RESPONSES,
    distance_bucket,
)
from boss_env.env import BossFightEnv, N_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _feed_observations(model: PlayerModel, n: int, ability_id: int = 1,
                        dist: float = 80.0, response: str = "roll") -> None:
    """Feed n identical observations into the model."""
    for _ in range(n):
        model.observe(ability_id, dist, response)


# ---------------------------------------------------------------------------
# Test 1: counts update correctly and tendencies sum sensibly
# ---------------------------------------------------------------------------

def test_counts_and_tendencies():
    model = PlayerModel()
    model.set_fight(1)

    # Initially zero samples -> uniform tendencies
    tend = model.tendencies()
    assert set(tend.keys()) == set(RESPONSES), "Tendencies must cover all RESPONSES"
    total = sum(tend.values())
    assert abs(total - 1.0) < 1e-9, f"Empty tendencies should sum to 1, got {total}"

    # Feed 10 rolls and 5 jumps for ability 1
    for _ in range(10):
        model.observe(1, 80.0, "roll")
    for _ in range(5):
        model.observe(1, 80.0, "jump")

    assert model.total_samples == 15, f"Expected 15 samples, got {model.total_samples}"

    tend2 = model.tendencies()
    total2 = sum(tend2.values())
    assert abs(total2 - 1.0) < 1e-9, f"Non-empty tendencies should sum to 1, got {total2}"
    assert tend2["roll"] > tend2["jump"], "Roll should dominate jump after 10 vs 5 obs"
    assert abs(tend2["roll"] - 10 / 15) < 1e-9, "Roll rate should be 10/15"

    # Feed some observations for ability 4 in mid range
    for _ in range(3):
        model.observe(4, 200.0, "parry")
    assert model.total_samples == 18

    tend3 = model.tendencies()
    assert abs(sum(tend3.values()) - 1.0) < 1e-9, "Tendencies must always sum to 1"

    print("TEST 1 PASSED: counts update correctly and tendencies sum to 1")


# ---------------------------------------------------------------------------
# Test 2: weights sum to 1; alpha never exceeds ALPHA_MAX
# ---------------------------------------------------------------------------

def test_weights_and_alpha():
    model = PlayerModel()
    model.set_fight(APPLY_FROM_FIGHT)  # weights active from here

    # Alpha should be 0 with no samples -> weights_for_candidates returns None
    candidates = [1, 2, 3, 4]
    w = model.weights_for_candidates(candidates, fight=APPLY_FROM_FIGHT)
    assert w is None, "No samples -> weights should be None (uniform fallback)"
    assert model.alpha == 0.0

    # Feed exactly SAMPLES_REF observations
    _feed_observations(model, SAMPLES_REF, ability_id=1, dist=80.0, response="jump")
    assert abs(model.alpha - ALPHA_MAX) < 1e-9, \
        f"After SAMPLES_REF obs alpha should be ALPHA_MAX, got {model.alpha}"

    # Feed more: alpha must not exceed ALPHA_MAX
    _feed_observations(model, SAMPLES_REF * 5, ability_id=1, dist=80.0, response="jump")
    assert model.alpha <= ALPHA_MAX + 1e-9, \
        f"Alpha exceeded ALPHA_MAX: {model.alpha} > {ALPHA_MAX}"

    # Weights for a multi-candidate list should sum to 1
    w2 = model.weights_for_candidates([1, 2, 4, 7], fight=APPLY_FROM_FIGHT)
    assert w2 is not None, "With ALPHA_MAX samples, weights should not be None"
    assert len(w2) == 4, "One weight per candidate"
    assert abs(sum(w2) - 1.0) < 1e-9, f"Weights must sum to 1, got {sum(w2)}"
    assert all(w >= 0 for w in w2), "All weights must be non-negative"

    # Weights before APPLY_FROM_FIGHT should be None
    w_early = model.weights_for_candidates([1, 2], fight=APPLY_FROM_FIGHT - 1)
    assert w_early is None, "Weights before APPLY_FROM_FIGHT should be None"

    # Single candidate -> None (no choice to make)
    w_single = model.weights_for_candidates([1], fight=APPLY_FROM_FIGHT)
    assert w_single is None, "Single candidate -> no weighting needed"

    print("TEST 2 PASSED: weights sum to 1; alpha bounded by ALPHA_MAX")


# ---------------------------------------------------------------------------
# Test 3: model freezes after LEARN_UNTIL_FIGHT
# ---------------------------------------------------------------------------

def test_model_freezes():
    model = PlayerModel()

    # Collect 5 observations before freeze fight
    model.set_fight(LEARN_UNTIL_FIGHT)
    for _ in range(5):
        model.observe(1, 80.0, "roll")
    assert model.total_samples == 5

    # Advance past the freeze fight
    model.set_fight(LEARN_UNTIL_FIGHT + 1)
    for _ in range(100):
        model.observe(1, 80.0, "parry")  # should be ignored

    assert model.total_samples == 5, \
        f"Model should be frozen after fight {LEARN_UNTIL_FIGHT}, got {model.total_samples}"

    # Even with a different response the count stays at 5
    model.observe(2, 200.0, "jump")
    assert model.total_samples == 5, "Model must not accept observations after freeze fight"

    print(f"TEST 3 PASSED: model freezes after fight {LEARN_UNTIL_FIGHT}")


# ---------------------------------------------------------------------------
# Test 4: regression – adaptive_boss=False gives same ability sequence
# ---------------------------------------------------------------------------

def test_regression_off_flag():
    """With adaptive_boss=False the ability sequence must equal the pre-patch baseline."""
    SEED = 42
    N_STEPS = 500
    FIGHT = 12   # fight 12 has 4 candidates (Lightning, Spikes, Wave, Bolt)

    def _run_env(attach_model: bool) -> list:
        """Run N_STEPS steps and collect boss ability selections."""
        env = BossFightEnv(weapon="katana", fight=FIGHT, frame_skip=1)
        if attach_model:
            m = PlayerModel()
            m.set_fight(FIGHT)
            env.set_player_model(m)
        obs, _ = env.reset(seed=SEED)
        abilities = []
        for _ in range(N_STEPS):
            # Always use IDLE to keep player position and timing deterministic
            obs, _, term, trunc, _ = env.step(0)
            abilities.append(env.boss.current_ability_id)
            if term or trunc:
                obs, _ = env.reset()
        return abilities

    seq_off = _run_env(attach_model=False)
    seq_off2 = _run_env(attach_model=False)

    # Baseline must itself be deterministic
    assert seq_off == seq_off2, "Non-adaptive env must be deterministic"

    # A model with ZERO samples must produce the same sequence because
    # weights_for_candidates returns None (alpha=0) -> uniform pick
    seq_with_empty_model = _run_env(attach_model=True)
    assert seq_off == seq_with_empty_model, (
        "Empty PlayerModel (alpha=0) must not change ability selection sequence"
    )

    print("TEST 4 PASSED: regression – adaptive_boss=False / empty model is identical to baseline")


# ---------------------------------------------------------------------------
# Test 5: JSON round-trip
# ---------------------------------------------------------------------------

def test_json_roundtrip():
    model = PlayerModel()
    model.set_fight(5)
    for _ in range(8):
        model.observe(1, 80.0, "roll")
    for _ in range(4):
        model.observe(4, 250.0, "jump")
    for _ in range(2):
        model.observe(7, 150.0, "parry")

    original_samples = model.total_samples
    original_fight   = model._current_fight
    original_counts  = dict(model._counts)

    json_str = model.to_json()
    model2 = PlayerModel.from_json(json_str)

    assert model2.total_samples == original_samples, \
        f"Round-trip samples mismatch: {model2.total_samples} != {original_samples}"
    assert model2._current_fight == original_fight, \
        f"Round-trip fight mismatch: {model2._current_fight} != {original_fight}"

    for ctx, counts in original_counts.items():
        assert ctx in model2._counts, f"Context {ctx} missing after round-trip"
        for resp, n in counts.items():
            assert model2._counts[ctx][resp] == n, \
                f"Count mismatch at {ctx}[{resp}]: {model2._counts[ctx][resp]} != {n}"

    # Tendencies must be identical
    t1 = model.tendencies()
    t2 = model2.tendencies()
    for r in RESPONSES:
        assert abs(t1[r] - t2[r]) < 1e-9, \
            f"Tendency mismatch for {r}: {t1[r]} != {t2[r]}"

    # Double round-trip
    model3 = PlayerModel.from_json(model2.to_json())
    assert model3.total_samples == original_samples

    print("TEST 5 PASSED: JSON round-trip is lossless")


# ---------------------------------------------------------------------------
# Additional: distance_bucket helper
# ---------------------------------------------------------------------------

def test_distance_bucket():
    assert distance_bucket(0.0)   == "close"
    assert distance_bucket(99.9)  == "close"
    assert distance_bucket(100.0) == "mid"
    assert distance_bucket(300.0) == "mid"
    assert distance_bucket(300.1) == "far"
    assert distance_bucket(800.0) == "far"
    print("TEST distance_bucket PASSED")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("ADAPTIVE BOSS – UNIT TESTS")
    print("=" * 60)

    test_distance_bucket()
    test_counts_and_tendencies()
    test_weights_and_alpha()
    test_model_freezes()
    test_regression_off_flag()
    test_json_roundtrip()

    print("\n" + "=" * 60)
    print("ALL ADAPTATION TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
