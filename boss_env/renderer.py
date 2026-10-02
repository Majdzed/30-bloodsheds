"""Pygame renderer for Boss Gauntlet.

Renders player, boss, projectiles, telegraphs, and full HUD.
Only reads env state; never modifies it.
"""

import math
import pygame
from . import constants as C
from .entities import PlayerState, BossState


class Renderer:
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Boss Gauntlet: 30 Bloodsheds")
        self.clock = pygame.time.Clock()

        # Fonts
        self.font_title = pygame.font.SysFont("georgia", 28, bold=True)
        self.font_main = pygame.font.SysFont("arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("arial", 14)
        self.font_hud = pygame.font.SysFont("consolas", 16, bold=True)

        # Arena mapping
        self.scale = 1.25  # 800 * 1.25 = 1000 px
        self.arena_left = (self.width - int(C.ARENA_W * self.scale)) // 2  # 140 px
        self.arena_right = self.arena_left + int(C.ARENA_W * self.scale)   # 1140 px
        self.ground_y = 540

    def world_to_screen(self, x: float, y: float = 0.0):
        sx = int(self.arena_left + x * self.scale)
        sy = int(self.ground_y - y * self.scale)
        return sx, sy

    def draw(self, env):
        p = env.player
        b = env.boss
        if p is None or b is None:
            return

        # 1. Background (Moonlit ancient courtyard)
        self.screen.fill((12, 14, 24))

        # Distant moon
        pygame.draw.circle(self.screen, (230, 235, 245), (1020, 130), 55)
        pygame.draw.circle(self.screen, (12, 14, 24), (1050, 120), 50)  # crescent shadow

        # Distant gothic skyline silhouette
        city_color = (20, 22, 38)
        pygame.draw.rect(self.screen, city_color, (0, 360, self.width, 180))
        # Towers
        for tx, tw, th in [(80, 70, 160), (220, 50, 110), (340, 90, 140), (860, 60, 130), (1080, 80, 170)]:
            pygame.draw.rect(self.screen, city_color, (tx, 360 - th, tw, th))
            pygame.draw.polygon(self.screen, city_color, [(tx, 360 - th), (tx + tw // 2, 360 - th - 40), (tx + tw, 360 - th)])

        # Courtyard floor
        floor_rect = pygame.Rect(0, self.ground_y, self.width, self.height - self.ground_y)
        pygame.draw.rect(self.screen, (32, 35, 48), floor_rect)
        pygame.draw.line(self.screen, (60, 65, 85), (0, self.ground_y), (self.width, self.ground_y), 3)

        # Arena boundary pillars
        pillar_w = 20
        pygame.draw.rect(self.screen, (55, 60, 80), (self.arena_left - pillar_w, 200, pillar_w, self.ground_y - 200))
        pygame.draw.rect(self.screen, (55, 60, 80), (self.arena_right, 200, pillar_w, self.ground_y - 200))
        # Wall highlight lines
        pygame.draw.line(self.screen, (100, 110, 140), (self.arena_left, 200), (self.arena_left, self.ground_y), 2)
        pygame.draw.line(self.screen, (100, 110, 140), (self.arena_right, 200), (self.arena_right, self.ground_y), 2)

        # 2. Boss Rendering
        bsx, bsy = self.world_to_screen(b.x, 0.0)
        bw, bh = 54, 96
        boss_rect = pygame.Rect(bsx - bw // 2, bsy - bh, bw, bh)

        # Boss telegraph aura / windup
        if b.state == BossState.WINDUP:
            aura_radius = int(60 + 10 * math.sin(pygame.time.get_ticks() * 0.02))
            aura_surf = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(aura_surf, (255, 50, 50, 70), (aura_radius, aura_radius), aura_radius)
            self.screen.blit(aura_surf, (bsx - aura_radius, bsy - bh // 2 - aura_radius))
            # Windup telegraph arc on ground for Lightning Strike
            if b.current_ability_id == 1:
                arc_w = int(120 * self.scale)
                arc_dir = 1 if p.x > b.x else -1
                arc_rect = pygame.Rect(bsx if arc_dir > 0 else bsx - arc_w, bsy - 20, arc_w, 20)
                pygame.draw.rect(self.screen, (255, 60, 60, 100), arc_rect, border_radius=4)

        # Boss color
        if b.state == BossState.STUN:
            b_color = (110, 110, 125)
        elif b.state == BossState.ATTACK:
            b_color = (230, 40, 60)
        elif b.state == BossState.WINDUP:
            b_color = (190, 50, 80)
        else:
            b_color = (130, 40, 150)

        pygame.draw.rect(self.screen, b_color, boss_rect, border_radius=6)
        pygame.draw.rect(self.screen, (20, 10, 25), boss_rect, width=2, border_radius=6)

        # Boss eyes
        eye_dir = 1 if p.x > b.x else -1
        eye_x = bsx + eye_dir * 12
        pygame.draw.circle(self.screen, (255, 230, 80), (eye_x, bsy - bh + 25), 4)

        # Boss Attack visual slash
        if b.state == BossState.ATTACK and b.current_ability_id == 1:
            slash_len = int(120 * self.scale)
            slash_dir = 1 if p.x > b.x else -1
            slash_color = (255, 220, 100)
            pygame.draw.arc(
                self.screen,
                slash_color,
                pygame.Rect(bsx - slash_len if slash_dir < 0 else bsx, bsy - bh, slash_len, bh),
                0 if slash_dir > 0 else math.pi / 2,
                math.pi / 2 if slash_dir > 0 else math.pi,
                6,
            )

        # 3. Player Rendering
        psx, psy = self.world_to_screen(p.x, p.y)
        pw, ph = 38, 74
        player_rect = pygame.Rect(psx - pw // 2, psy - ph, pw, ph)

        # Invulnerability aura (Roll / Super)
        if p.iframes > 0:
            glow_surf = pygame.Surface((pw + 20, ph + 20), pygame.SRCALPHA)
            glow_color = (255, 60, 80, 120) if p.state == PlayerState.SUPER else (130, 220, 255, 100)
            pygame.draw.rect(glow_surf, glow_color, (0, 0, pw + 20, ph + 20), border_radius=10)
            self.screen.blit(glow_surf, (psx - pw // 2 - 10, psy - ph - 10))

        # Player color by state
        if p.state == PlayerState.ROLL:
            p_color = (100, 200, 255)
        elif p.state == PlayerState.PARRY:
            p_color = (255, 225, 70)
        elif p.state == PlayerState.SUPER:
            p_color = (240, 30, 60)
        elif p.state == PlayerState.ATTACK:
            p_color = (255, 150, 40)
        else:
            p_color = (70, 140, 230)

        pygame.draw.rect(self.screen, p_color, player_rect, border_radius=5)
        pygame.draw.rect(self.screen, (15, 25, 50), player_rect, width=2, border_radius=5)

        # Player weapon slash visual
        if p.state == PlayerState.ATTACK:
            range_u = env.weapon.light_range if p.pending_attack_id == 1 else env.weapon.heavy_range
            slash_len = int(range_u * self.scale)
            slash_dir = 1 if b.x > p.x else -1
            end_x = psx + slash_dir * slash_len
            sword_color = (255, 240, 180) if p.pending_attack_id == 2 else (200, 230, 255)
            pygame.draw.line(self.screen, sword_color, (psx, psy - ph // 2), (end_x, psy - ph // 2 - 10), 5)

        # 4. Projectiles (Light Bolt)
        for bolt in env._light_bolts:
            bx, by = self.world_to_screen(bolt["x"], 35.0)
            b_dir = bolt["dir"]
            # Draw glowing golden comet
            pygame.draw.ellipse(self.screen, (255, 235, 120), (bx - 14, by - 6, 28, 12))
            pygame.draw.circle(self.screen, (255, 255, 230), (int(bx + b_dir * 8), int(by)), 5)
            # Trail
            tail_x = int(bx - b_dir * 30)
            pygame.draw.line(self.screen, (255, 190, 60), (bx, by), (tail_x, by), 3)

        # 5. Full HUD Interface
        self._draw_hud(env)

        pygame.display.flip()
        self.clock.tick(C.FPS)

    def _draw_hud(self, env):
        p = env.player
        b = env.boss

        # --- Boss HP Bar (Top Center) ---
        bar_w, bar_h = 500, 24
        bar_x = (self.width - bar_w) // 2
        bar_y = 35

        # Boss Name and Ability banner
        boss_title = f"VON VASTORIOUS (Fight {env.fight}/30)"
        txt_boss = self.font_main.render(boss_title, True, (240, 220, 230))
        self.screen.blit(txt_boss, (bar_x, bar_y - 24))

        if b.state == BossState.WINDUP:
            ab_name = "LIGHTNING STRIKE" if b.current_ability_id == 1 else "LIGHT BOLT"
            txt_warn = self.font_main.render(f"! {ab_name} !", True, (255, 70, 70))
            self.screen.blit(txt_warn, (bar_x + bar_w - txt_warn.get_width(), bar_y - 24))
        elif b.state == BossState.STUN:
            txt_stun = self.font_main.render("STUNNED!", True, (255, 220, 80))
            self.screen.blit(txt_stun, (bar_x + bar_w - txt_stun.get_width(), bar_y - 24))

        # HP Bar Background & Fill
        pygame.draw.rect(self.screen, (30, 20, 25), (bar_x - 2, bar_y - 2, bar_w + 4, bar_h + 4), border_radius=4)
        hp_frac = max(0.0, b.hp / b.max_hp)
        fill_w = int(bar_w * hp_frac)
        if fill_w > 0:
            pygame.draw.rect(self.screen, (190, 30, 50), (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        pygame.draw.rect(self.screen, (120, 70, 80), (bar_x, bar_y, bar_w, bar_h), 2, border_radius=3)

        txt_hp = self.font_hud.render(f"{int(b.hp)} / {int(b.max_hp)}", True, (255, 255, 255))
        self.screen.blit(txt_hp, (bar_x + (bar_w - txt_hp.get_width()) // 2, bar_y + 4))

        # --- Player HUD (Top Left) ---
        hud_x, hud_y = 40, 35

        # Player Hearts
        for i in range(p.max_hearts):
            hx = hud_x + i * 32
            hy = hud_y
            if i < p.hearts:
                # Full red heart
                pygame.draw.circle(self.screen, (230, 40, 50), (hx + 8, hy + 8), 7)
                pygame.draw.circle(self.screen, (230, 40, 50), (hx + 20, hy + 8), 7)
                pygame.draw.polygon(self.screen, (230, 40, 50), [(hx + 1, hy + 10), (hx + 27, hy + 10), (hx + 14, hy + 24)])
            else:
                # Empty heart frame
                pygame.draw.circle(self.screen, (70, 70, 80), (hx + 8, hy + 8), 7, 2)
                pygame.draw.circle(self.screen, (70, 70, 80), (hx + 20, hy + 8), 7, 2)
                pygame.draw.polygon(self.screen, (70, 70, 80), [(hx + 1, hy + 10), (hx + 27, hy + 10), (hx + 14, hy + 24)], 2)

        # Stamina Bar
        stam_w, stam_h = 160, 10
        stam_y = hud_y + 34
        pygame.draw.rect(self.screen, (25, 30, 35), (hud_x, stam_y, stam_w, stam_h), border_radius=2)
        stam_fill = int(stam_w * (p.stamina / C.STAMINA_MAX))
        if stam_fill > 0:
            pygame.draw.rect(self.screen, (40, 200, 140), (hud_x, stam_y, stam_fill, stam_h), border_radius=2)
        pygame.draw.rect(self.screen, (80, 110, 100), (hud_x, stam_y, stam_w, stam_h), 1, border_radius=2)

        # Super Meter (3 Stacks)
        meter_y = stam_y + 18
        txt_super = self.font_small.render("SUPER:", True, (180, 190, 210))
        self.screen.blit(txt_super, (hud_x, meter_y))
        for s in range(C.SUPER_STACKS_REQUIRED):
            sx = hud_x + 60 + s * 24
            sy = meter_y + 7
            is_full = s < p.super_stacks
            color = (255, 50, 70) if is_full else (60, 65, 80)
            pts = [(sx, sy - 8), (sx + 8, sy), (sx, sy + 8), (sx - 8, sy)]
            pygame.draw.polygon(self.screen, color, pts)
            pygame.draw.polygon(self.screen, (160, 170, 190), pts, 1)

        # Flasks Count
        flask_y = meter_y + 24
        txt_flasks = self.font_small.render(f"FLASKS: {p.flasks}", True, (80, 210, 120))
        self.screen.blit(txt_flasks, (hud_x, flask_y))

        # --- Timer (Top Right) ---
        time_left = max(0.0, C.FIGHT_TIME_LIMIT_S - (env._fight_frame / C.FPS))
        time_color = (255, 80, 80) if time_left < 15.0 else (220, 230, 240)
        txt_time = self.font_hud.render(f"TIME: {time_left:05.1f}s", True, time_color)
        self.screen.blit(txt_time, (self.width - txt_time.get_width() - 40, 35))

        # --- Controls Reference (Bottom Center) ---
        ctrl_text = "A/D: Move | W/Space: Jump | J: Light Attack | K: Heavy Attack | L: Roll | U: Parry | I: Blood Katana | R: Reset"
        txt_ctrl = self.font_small.render(ctrl_text, True, (130, 140, 160))
        self.screen.blit(txt_ctrl, ((self.width - txt_ctrl.get_width()) // 2, self.height - 30))

    def close(self):
        pygame.quit()
