"""Custom feature extractor (required: no predefined SB3 architecture).

TODO: subclass stable_baselines3.common.torch_layers.BaseFeaturesExtractor.
Idea: separate small encoders for player / boss / hazard groups, then merge.
"""
