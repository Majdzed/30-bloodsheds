"""Custom feature extractor for Boss Gauntlet RL agent.

Per SPEC.md Section 8:
"Algorithm: MaskablePPO with a custom features extractor.
Idea: separate small encoders for player / boss / hazard groups, then merge."
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class BossGauntletFeaturesExtractor(BaseFeaturesExtractor):
    """Hierarchical structured feature extractor for the 52-dimensional observation space.

    Decomposes the observation into domain-specific streams:
    1. Player Stream (21 dims): 14 scalars + 7 one-hot player states -> Linear(21, 64) -> ReLU
    2. Boss Stream (18 dims): 5 scalars + 5 one-hot boss states + 8 one-hot attacks -> Linear(18, 64) -> ReLU
    3. Hazard Stream (5 dims): 5 hazard scalars (spikes, projectiles, orb parry) -> Linear(5, 32) -> ReLU
    4. Context Stream (8 dims): fight index + 7 ability unlock bits -> Linear(8, 32) -> ReLU

    Merged Representation:
    Concat(64 + 64 + 32 + 32 = 192) -> Linear(192, features_dim) -> ReLU
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # Slice definitions based on SPEC.md and env.py
        # Scalars (32 total):
        #   [0:14]   Player scalars (14)
        #   [14:19]  Boss scalars (5)
        #   [19:24]  Hazard scalars (5)
        #   [24:32]  Context scalars (8)
        # One-hots (20 total):
        #   [32:39]  Player state (7)
        #   [39:44]  Boss state (5)
        #   [44:52]  Boss attack (8)

        # Player encoder: 14 scalars + 7 state one-hot = 21
        self.player_net = nn.Sequential(
            nn.Linear(14 + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Boss encoder: 5 scalars + 5 state one-hot + 8 attack one-hot = 18
        self.boss_net = nn.Sequential(
            nn.Linear(5 + 5 + 8, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Hazard encoder: 5 hazard scalars
        self.hazard_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
        )

        # Context encoder: 8 context scalars
        self.context_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
        )

        # Fusion layer
        total_stream_dim = 64 + 64 + 32 + 32  # 192
        self.fusion_net = nn.Sequential(
            nn.Linear(total_stream_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 1. Slice observation groups
        p_scalars = observations[:, 0:14]
        b_scalars = observations[:, 14:19]
        h_scalars = observations[:, 19:24]
        c_scalars = observations[:, 24:32]

        p_state = observations[:, 32:39]
        b_state = observations[:, 39:44]
        b_attack = observations[:, 44:52]

        # 2. Concat into stream inputs
        player_in = torch.cat([p_scalars, p_state], dim=1)
        boss_in = torch.cat([b_scalars, b_state, b_attack], dim=1)
        hazard_in = h_scalars
        context_in = c_scalars

        # 3. Process streams
        p_feat = self.player_net(player_in)
        b_feat = self.boss_net(boss_in)
        h_feat = self.hazard_net(hazard_in)
        c_feat = self.context_net(context_in)

        # 4. Fuse representations
        merged = torch.cat([p_feat, b_feat, h_feat, c_feat], dim=1)
        return self.fusion_net(merged)
