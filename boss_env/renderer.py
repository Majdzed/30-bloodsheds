"""Pygame renderer for Boss Gauntlet.

Renders player, boss, projectiles, ability telegraphs, and full HUD.
Only reads env state; never modifies it.
"""

import os
import math
import pygame
from . import constants as C
from .abilities import ABILITIES
from .entities import PlayerState, BossState


class NightBorneBossAnimator:
    """Manages sprite frame slicing and animation state for the NightBorne boss."""
    def __init__(self, sprite_path: str = "assets/NightBorne.png", scale: float = 3.0):
        self.scale = scale
        self.frame_w = 80
        self.frame_h = 80
        self.scaled_w = int(self.frame_w * scale)
        self.scaled_h = int(self.frame_h * scale)
        self.feet_y_offset = int(63 * scale)    # Baseline where feet touch the ground (y ≈ 63 in 80px cell)
        self.center_x_offset = int(36 * scale)  # Visual center of character body (x ≈ 36 in 80px cell)

        self.animations = {
            "idle": [],
            "run": [],
            "attack": [],
            "hurt": [],
            "death": []
        }
        self.loaded = False
        self._load_spritesheet(sprite_path)

    def _load_spritesheet(self, path: str):
        if not os.path.exists(path):
            return
        try:
            sheet = pygame.image.load(path).convert_alpha()
            rows_config = [
                ("idle", 0, 9),
                ("run", 1, 6),
                ("attack", 2, 12),
                ("hurt", 3, 5),
                ("death", 4, 23),
            ]
            for name, row, count in rows_config:
                frames = []
                for col in range(count):
                    rect = pygame.Rect(col * self.frame_w, row * self.frame_h, self.frame_w, self.frame_h)
                    sub = sheet.subsurface(rect)
                    scaled = pygame.transform.scale(sub, (self.scaled_w, self.scaled_h))
                    frames.append(scaled)
                self.animations[name] = frames
            self.loaded = True
        except Exception as e:
            print(f"[Renderer] Could not load NightBorne sprite: {e}")
            self.loaded = False

    def get_frame_and_rect(self, boss, player_x: float, tick: int, death_tick: int, windup_total: int, bsx: int, bsy: int):
        """Returns (surface, (draw_x, draw_y)) for current boss state."""
        if not self.loaded:
            return None, (0, 0)

        facing_right = player_x >= boss.x

        if boss.hp <= 0:
            frames = self.animations["death"]
            idx = min(len(frames) - 1, death_tick // 4)
            frame = frames[idx]
        elif boss.state == BossState.STUN:
            frames = self.animations["hurt"]
            idx = (tick // 6) % len(frames)
            frame = frames[idx]
        elif boss.state == BossState.WINDUP:
            frames = self.animations["attack"]
            # Attack frames 0..6 represent the sword raising windup
            wt = max(1, windup_total)
            progress = 1.0 - (boss.action_timer / wt)
            idx = min(6, int(progress * 7))
            frame = frames[idx]
        elif boss.state == BossState.ATTACK:
            frames = self.animations["attack"]
            # Attack frames 7..11 are the slash and recovery follow-through
            idx = 7 + min(4, (tick // 3) % 5)
            frame = frames[idx]
        elif boss.state == BossState.HEAL:
            # Siphon Heal: casting posture with raised blade
            frames = self.animations["attack"]
            frame = frames[4]
        else:
            # Idle vs Movement
            dist_to_player = abs(player_x - boss.x)
            if dist_to_player > 100.0:
                frames = self.animations["run"]
                idx = (tick // 5) % len(frames)
            else:
                frames = self.animations["idle"]
                idx = (tick // 6) % len(frames)
            frame = frames[idx]

        # NightBorne faces RIGHT natively; flip if facing left
        if not facing_right:
            frame = pygame.transform.flip(frame, True, False)
            draw_x = bsx - (self.scaled_w - self.center_x_offset)
        else:
            draw_x = bsx - self.center_x_offset

        draw_y = bsy - self.feet_y_offset
        return frame, (draw_x, draw_y)


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

        # Sprite animators
        self.boss_animator = NightBorneBossAnimator()
        self.anim_tick = 0
        self.boss_death_timer = 0

    def world_to_screen(self, x: float, y: float = 0.0):
        sx = int(self.arena_left + x * self.scale)
        sy = int(self.ground_y - y * self.scale)
        return sx, sy

    def draw(self, env):
        if hasattr(env, "unwrapped"):
            env = env.unwrapped
        p = env.player
        b = env.boss
        if p is None or b is None:
            return

        now_ticks = pygame.time.get_ticks()
        self.anim_tick += 1
        if b.hp <= 0:
            self.boss_death_timer += 1
        else:
            self.boss_death_timer = 0

        # -------------------------------------------------------------
        # 1. Background (Moonlit gothic courtyard)
        # -------------------------------------------------------------
        self.screen.fill((12, 14, 24))

        # Distant moon
        pygame.draw.circle(self.screen, (230, 235, 245), (1020, 130), 55)
        pygame.draw.circle(self.screen, (12, 14, 24), (1050, 120), 50)  # crescent shadow

        # Gothic skyline silhouette
        city_color = (20, 22, 38)
        pygame.draw.rect(self.screen, city_color, (0, 360, self.width, 180))
        for tx, tw, th in [(80, 70, 160), (220, 50, 110), (340, 90, 140), (860, 60, 130), (1080, 80, 170)]:
            pygame.draw.rect(self.screen, city_color, (tx, 360 - th, tw, th))
            pygame.draw.polygon(self.screen, city_color, [(tx, 360 - th), (tx + tw // 2, 360 - th - 40), (tx + tw, 360 - th)])

        # Floor
        floor_rect = pygame.Rect(0, self.ground_y, self.width, self.height - self.ground_y)
        pygame.draw.rect(self.screen, (32, 35, 48), floor_rect)
        pygame.draw.line(self.screen, (60, 65, 85), (0, self.ground_y), (self.width, self.ground_y), 3)

        # Arena boundary pillars
        pillar_w = 20
        pygame.draw.rect(self.screen, (55, 60, 80), (self.arena_left - pillar_w, 200, pillar_w, self.ground_y - 200))
        pygame.draw.rect(self.screen, (55, 60, 80), (self.arena_right, 200, pillar_w, self.ground_y - 200))
        pygame.draw.line(self.screen, (100, 110, 140), (self.arena_left, 200), (self.arena_left, self.ground_y), 2)
        pygame.draw.line(self.screen, (100, 110, 140), (self.arena_right, 200), (self.arena_right, self.ground_y), 2)

        # -------------------------------------------------------------
        # 2. Hazards & Boss Ability Ground Telegraphs
        # -------------------------------------------------------------
        # 2A. Dark Spikes (Ability 2)
        for spike in getattr(env, "_dark_spikes", []):
            sp_x, sp_y = self.world_to_screen(spike["x"], 0.0)
            sp_radius = int(70 * self.scale)  # 140u diameter
            if spike.get("warning_timer", 0) > 0:
                spike_surf = pygame.Surface((sp_radius * 2, 30), pygame.SRCALPHA)
                alpha = int(120 + 80 * math.sin(now_ticks * 0.03))
                pygame.draw.ellipse(spike_surf, (180, 30, 220, alpha), (0, 0, sp_radius * 2, 30))
                self.screen.blit(spike_surf, (sp_x - sp_radius, sp_y - 15))
            elif spike.get("active_timer", 0) > 0:
                # Spikes erupting from the ground!
                for ox in range(-sp_radius + 10, sp_radius - 10, 20):
                    spike_h = int(60 + 20 * math.sin(ox))
                    pygame.draw.polygon(
                        self.screen,
                        (220, 40, 240),
                        [(sp_x + ox - 8, sp_y), (sp_x + ox, sp_y - spike_h), (sp_x + ox + 8, sp_y)]
                    )

        # -------------------------------------------------------------
        # 3. Boss Rendering
        # -------------------------------------------------------------
        bsx, bsy = self.world_to_screen(b.x, 0.0)
        bw, bh = 54, 96
        boss_rect = pygame.Rect(bsx - bw // 2, bsy - bh, bw, bh)

        # Windup aura
        if b.state == BossState.WINDUP:
            aura_radius = int(60 + 10 * math.sin(now_ticks * 0.02))
            aura_surf = pygame.Surface((aura_radius * 2, aura_radius * 2), pygame.SRCALPHA)
            color = (255, 50, 50, 70) if b.current_ability_id in (1, 7) else (180, 50, 255, 70)
            pygame.draw.circle(aura_surf, color, (aura_radius, aura_radius), aura_radius)
            self.screen.blit(aura_surf, (bsx - aura_radius, bsy - bh // 2 - aura_radius))
            # Melee warning arc
            if b.current_ability_id in (1, 7):
                arc_w = int(120 * self.scale)
                arc_dir = 1 if p.x > b.x else -1
                arc_rect = pygame.Rect(bsx if arc_dir > 0 else bsx - arc_w, bsy - 15, arc_w, 15)
                pygame.draw.rect(self.screen, (255, 60, 60, 110), arc_rect, border_radius=4)

        if self.boss_animator.loaded:
            windup_total = getattr(env, "_boss_windup_total", 30)
            frame, (draw_x, draw_y) = self.boss_animator.get_frame_and_rect(
                boss=b,
                player_x=p.x,
                tick=self.anim_tick,
                death_tick=self.boss_death_timer,
                windup_total=windup_total,
                bsx=bsx,
                bsy=bsy,
            )

            # Boss Stun visual indicator (dizzy stars orbiting above head)
            if b.state == BossState.STUN:
                for star_i in range(3):
                    star_angle = (now_ticks * 0.008) + star_i * (2 * math.pi / 3)
                    star_x = bsx + int(24 * math.cos(star_angle))
                    star_y = bsy - int(bh * 1.1) + int(8 * math.sin(star_angle))
                    pygame.draw.circle(self.screen, (255, 230, 80), (star_x, star_y), 3)

            # Boss Siphon Heal aura (pulsing green siphon circle)
            if b.state == BossState.HEAL:
                heal_surf = pygame.Surface((120, 120), pygame.SRCALPHA)
                alpha = int(70 + 40 * math.sin(now_ticks * 0.02))
                pygame.draw.circle(heal_surf, (50, 230, 120, alpha), (60, 60), 50)
                self.screen.blit(heal_surf, (bsx - 60, bsy - 100))

            if frame is not None:
                self.screen.blit(frame, (draw_x, draw_y))
        else:
            # Fallback to geometric rectangle
            if b.state == BossState.STUN:
                b_color = (110, 110, 125)
            elif b.state == BossState.ATTACK:
                b_color = (230, 40, 60)
            elif b.state == BossState.WINDUP:
                b_color = (190, 50, 80)
            elif b.state == BossState.HEAL:
                b_color = (50, 200, 120)
            else:
                b_color = (130, 40, 150)

            pygame.draw.rect(self.screen, b_color, boss_rect, border_radius=6)
            pygame.draw.rect(self.screen, (20, 10, 25), boss_rect, width=2, border_radius=6)

            # Boss eyes
            eye_dir = 1 if p.x > b.x else -1
            eye_x = bsx + eye_dir * 12
            eye_color = (255, 230, 80) if b.state != BossState.STUN else (70, 70, 70)
            pygame.draw.circle(self.screen, eye_color, (eye_x, bsy - bh + 25), 4)

            # Boss Melee / Frenzy slash visuals
            if b.state == BossState.ATTACK:
                slash_len = int(130 * self.scale)
                slash_dir = 1 if p.x > b.x else -1
                slash_color = (255, 80, 80) if b.current_ability_id == 7 else (255, 220, 100)
                pygame.draw.arc(
                    self.screen,
                    slash_color,
                    pygame.Rect(bsx - slash_len if slash_dir < 0 else bsx, bsy - bh, slash_len, bh),
                    0 if slash_dir > 0 else math.pi / 2,
                    math.pi / 2 if slash_dir > 0 else math.pi,
                    7 if b.current_ability_id == 7 else 5,
                )

        # -------------------------------------------------------------
        # 4. Projectiles and Special Hazards
        # -------------------------------------------------------------
        # 4A. Siphon Heal Orbs (Ability 5)
        for ox in env._heal_orbs:
            orb_sx, _ = self.world_to_screen(ox, 0.0)
            orb_sy = bsy - bh // 2 + int(18 * math.sin(now_ticks * 0.01 + ox))
            # Glowing green health sphere
            pygame.draw.circle(self.screen, (80, 255, 140), (orb_sx, orb_sy), 12)
            pygame.draw.circle(self.screen, (200, 255, 220), (orb_sx, orb_sy), 6)
            pygame.draw.circle(self.screen, (20, 120, 60), (orb_sx, orb_sy), 12, 2)

        # 4B. Dark Wave (Ability 3)
        for wave in env._dark_waves:
            wx, wy = self.world_to_screen(wave["x"], 0.0)
            wdir = wave["dir"]
            # Dark creeping ground wave
            wave_pts = [
                (wx - wdir * 25, wy),
                (wx, wy - 30),
                (wx + wdir * 15, wy),
            ]
            pygame.draw.polygon(self.screen, (160, 40, 220), wave_pts)
            pygame.draw.polygon(self.screen, (220, 120, 255), wave_pts, 2)

        # 4C. Light Bolt (Ability 4)
        for bolt in env._light_bolts:
            bx, by = self.world_to_screen(bolt["x"], 35.0)
            b_dir = bolt["dir"]
            pygame.draw.ellipse(self.screen, (255, 235, 120), (bx - 14, by - 6, 28, 12))
            pygame.draw.circle(self.screen, (255, 255, 230), (int(bx + b_dir * 8), int(by)), 5)
            tail_x = int(bx - b_dir * 30)
            pygame.draw.line(self.screen, (255, 190, 60), (bx, by), (tail_x, by), 3)

        # 4D. Ultimate Dark Orb (Ability 6)
        for orb in env._dark_orbs:
            ox, _ = self.world_to_screen(orb["x"], 0.0)
            # Arena-height towering sphere
            orb_radius = int(85 * self.scale)
            orb_surf = pygame.Surface((orb_radius * 2, self.height), pygame.SRCALPHA)
            pygame.draw.circle(orb_surf, (80, 10, 110, 200), (orb_radius, self.ground_y // 2), orb_radius)
            pygame.draw.circle(orb_surf, (220, 60, 255, 120), (orb_radius, self.ground_y // 2), orb_radius - 15)
            pygame.draw.circle(orb_surf, (255, 255, 255, 230), (orb_radius, self.ground_y // 2), 20)
            self.screen.blit(orb_surf, (ox - orb_radius, 0))

        # -------------------------------------------------------------
        # 5. Player Rendering
        # -------------------------------------------------------------
        psx, psy = self.world_to_screen(p.x, p.y)
        pw, ph = 38, 74
        player_rect = pygame.Rect(psx - pw // 2, psy - ph, pw, ph)

        # Invulnerability aura
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
        elif p.state == PlayerState.DRINK:
            p_color = (70, 220, 130)
        else:
            p_color = (70, 140, 230)

        pygame.draw.rect(self.screen, p_color, player_rect, border_radius=5)
        pygame.draw.rect(self.screen, (15, 25, 50), player_rect, width=2, border_radius=5)

        # Player attack visuals
        facing_boss = 1 if b.x > p.x else -1
        if p.state == PlayerState.ATTACK:
            if env.weapon.uses_ammo:
                # Revolver bullet tracers
                shot_y = psy - ph // 2 - 5
                if p.pending_attack_id == 2:
                    # Piercing heavy shot: fullscreen cyan beam
                    target_x = self.arena_right if facing_boss > 0 else self.arena_left
                    pygame.draw.line(self.screen, (100, 240, 255), (psx, shot_y), (target_x, shot_y), 6)
                    pygame.draw.line(self.screen, (255, 255, 255), (psx, shot_y), (target_x, shot_y), 2)
                else:
                    # Light attack bullet tracer
                    target_x = self.arena_right if facing_boss > 0 else self.arena_left
                    pygame.draw.line(self.screen, (255, 230, 100), (psx, shot_y), (target_x, shot_y), 3)
            else:
                # Katana sword slash
                range_u = env.weapon.light_range if p.pending_attack_id == 1 else env.weapon.heavy_range
                slash_len = int(range_u * self.scale)
                end_x = psx + facing_boss * slash_len
                sword_color = (255, 240, 180) if p.pending_attack_id == 2 else (200, 230, 255)
                pygame.draw.line(self.screen, sword_color, (psx, psy - ph // 2), (end_x, psy - ph // 2 - 10), 5)

        # Player Super visual
        if p.state == PlayerState.SUPER:
            if env.weapon.uses_ammo:
                # Ultra Plasma Shot: giant fullscreen laser
                beam_y = psy - ph // 2
                pygame.draw.line(self.screen, (255, 80, 120), (0, beam_y), (self.width, beam_y), 28)
                pygame.draw.line(self.screen, (255, 255, 255), (0, beam_y), (self.width, beam_y), 10)
            else:
                # Blood Katana: triple crimson slash arc
                slash_reach = int(200 * self.scale)
                arc_x = psx if facing_boss > 0 else psx - slash_reach
                pygame.draw.arc(self.screen, (255, 30, 60), pygame.Rect(arc_x, psy - ph - 20, slash_reach, ph + 40), 0, math.pi, 8)

        # -------------------------------------------------------------
        # 6. Full HUD Interface
        # -------------------------------------------------------------
        self._draw_hud(env)

        pygame.display.flip()
        self.clock.tick(C.FPS)

    def _draw_hud(self, env):
        p = env.player
        b = env.boss

        # --- Boss HP Bar (Top Center) ---
        bar_w, bar_h = 520, 24
        bar_x = (self.width - bar_w) // 2
        bar_y = 35

        boss_title = f"VON VASTORIOUS (Fight {env.fight}/30)"
        txt_boss = self.font_main.render(boss_title, True, (240, 220, 230))
        self.screen.blit(txt_boss, (bar_x, bar_y - 24))

        # Dynamic ability telegraph banner
        if b.state == BossState.WINDUP:
            ab_name = ABILITIES[b.current_ability_id - 1].name.upper() if 1 <= b.current_ability_id <= len(ABILITIES) else "ATTACK"
            txt_warn = self.font_main.render(f"! {ab_name} !", True, (255, 70, 70))
            self.screen.blit(txt_warn, (bar_x + bar_w - txt_warn.get_width(), bar_y - 24))
        elif b.state == BossState.HEAL:
            txt_heal = self.font_main.render("! SIPHON HEAL !", True, (80, 255, 140))
            self.screen.blit(txt_heal, (bar_x + bar_w - txt_heal.get_width(), bar_y - 24))
        elif b.state == BossState.STUN:
            txt_stun = self.font_main.render("STUNNED!", True, (255, 220, 80))
            self.screen.blit(txt_stun, (bar_x + bar_w - txt_stun.get_width(), bar_y - 24))

        # Boss HP bar
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
                pygame.draw.circle(self.screen, (230, 40, 50), (hx + 8, hy + 8), 7)
                pygame.draw.circle(self.screen, (230, 40, 50), (hx + 20, hy + 8), 7)
                pygame.draw.polygon(self.screen, (230, 40, 50), [(hx + 1, hy + 10), (hx + 27, hy + 10), (hx + 14, hy + 24)])
            else:
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

        # Revolver Ammo HUD (if Revolver)
        if env.weapon.uses_ammo:
            ammo_y = flask_y + 22
            txt_ammo = self.font_small.render("AMMO:", True, (240, 210, 90))
            self.screen.blit(txt_ammo, (hud_x, ammo_y))
            for b_idx in range(C.REVOLVER_MAG):
                bx = hud_x + 60 + b_idx * 16
                by = ammo_y + 2
                has_bullet = b_idx < p.ammo
                b_col = (255, 215, 60) if has_bullet else (60, 65, 75)
                pygame.draw.rect(self.screen, b_col, (bx, by, 10, 14), border_radius=2)
                pygame.draw.rect(self.screen, (160, 140, 50) if has_bullet else (40, 45, 55), (bx, by, 10, 14), 1, border_radius=2)

            if p.reload_timer > 0:
                rel_frac = 1.0 - (p.reload_timer / C.sec(C.REVOLVER_RELOAD_S))
                rel_bar_w = 90
                pygame.draw.rect(self.screen, (40, 45, 50), (hud_x + 60, ammo_y + 20, rel_bar_w, 6), border_radius=2)
                pygame.draw.rect(self.screen, (100, 220, 255), (hud_x + 60, ammo_y + 20, int(rel_bar_w * rel_frac), 6), border_radius=2)

        # --- Timer (Top Right) ---
        time_left = max(0.0, C.FIGHT_TIME_LIMIT_S - (env._fight_frame / C.FPS))
        time_color = (255, 80, 80) if time_left < 15.0 else (220, 230, 240)
        txt_time = self.font_hud.render(f"TIME: {time_left:05.1f}s", True, time_color)
        self.screen.blit(txt_time, (self.width - txt_time.get_width() - 40, 35))

        # --- Weapon Name & Controls Footer (Bottom Center) ---
        wpn_name = env.weapon.name
        super_name = "Blood Katana" if not env.weapon.uses_ammo else "Ultra Plasma"
        ctrl_text = f"[{wpn_name}] A/D: Move | W/Space: Jump | J: Light | K: Heavy | L: Roll | U: Parry | I: {super_name} | F: Flask"
        txt_ctrl = self.font_small.render(ctrl_text, True, (140, 150, 175))
        self.screen.blit(txt_ctrl, ((self.width - txt_ctrl.get_width()) // 2, self.height - 30))

    def close(self):
        pygame.quit()
