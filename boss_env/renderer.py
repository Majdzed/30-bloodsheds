"""Pygame drawing only. Reads env state, never changes it.

TODO: background (moonlit old city), player/boss sprites or shapes, hearts,
boss HP bar, ability telegraphs (warnings, projectiles, orbs), super meter.
Do NOT draw the agent-only features (fight index, unlocked mask).
"""


class Renderer:
    def __init__(self, width=1280, height=720):
        self.width = width
        self.height = height
        # TODO: pygame.init(), window, assets from assets/

    def draw(self, env):
        raise NotImplementedError

    def close(self):
        pass
