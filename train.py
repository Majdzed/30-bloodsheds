"""Train the RL agent across the 30-fight gauntlet.

TODO:
  - MaskablePPO (sb3-contrib) with the custom extractor from boss_env/networks.py
  - args: --weapon {katana,revolver} --upgrade {health,flask} --timesteps
  - advance env.set_fight() through 1..30, carrying weights forward
  - log undiscounted total reward per episode to TensorBoard (runs/)
  - save checkpoints to models/
"""

if __name__ == "__main__":
    raise SystemExit("train.py: not implemented yet")
