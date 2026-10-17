"""Interactive mode: Play or Watch the Boss Gauntlet with full menu navigation.

Features:
- Main Menu: Mode selection (Player vs Boss, Agent vs Boss, Train Agents, Quit)
- Character Selection: Hero Knight (Katana) or The Hunter (Revolver), fight selection 1..30
- In-Game Combat: Keyboard controls for player or autonomous AI control for agent
- Defeat Screen: Blood-drenched gothic theme with dripping blood streams, combat stats, retry/menu/quit
- Victory Screen: Radiant celebratory gold-crimson aura, stats, next fight progression
- Training Menu: Train single fight or full curriculum in a background thread without freezing UI
"""

import argparse
import os
import sys
import time
import threading
import pygame
import numpy as np

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action
from boss_env.adaptation import PlayerModel, LEARN_UNTIL_FIGHT
from boss_env.renderer import Renderer
from observe import find_latest_model
from agents.rule_based import RuleBasedAgent

try:
    from sb3_contrib import MaskablePPO
except ImportError:
    MaskablePPO = None


class GameState:
    MAIN_MENU = "MAIN_MENU"
    CHAR_SELECT = "CHAR_SELECT"
    TRAIN_MENU = "TRAIN_MENU"
    PLAYING = "PLAYING"
    GAME_OVER = "GAME_OVER"
    VICTORY = "VICTORY"


def get_action_from_keys(keys, env: BossFightEnv) -> Action:
    mask = env.action_masks()

    # 1. Super
    if (keys[pygame.K_i] or keys[pygame.K_q]) and mask[Action.USE_SUPER]:
        return Action.USE_SUPER

    # 2. Flask
    if (keys[pygame.K_h] or keys[pygame.K_e]) and mask[Action.USE_FLASK]:
        return Action.USE_FLASK

    # 3. Parry
    if (keys[pygame.K_u] or keys[pygame.K_f]) and mask[Action.PARRY]:
        return Action.PARRY

    # 4. Roll (hold A/Left to roll left, D/Right to roll right, neutral rolls away from boss)
    if (keys[pygame.K_l] or keys[pygame.K_c] or keys[pygame.K_LSHIFT]) and mask[Action.DODGE_ROLL]:
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            env.player.requested_roll_dir = 1.0
        elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
            env.player.requested_roll_dir = -1.0
        else:
            env.player.requested_roll_dir = 0.0
        return Action.DODGE_ROLL

    # 5. Attacks
    if (keys[pygame.K_k] or keys[pygame.K_x]) and mask[Action.HEAVY_ATTACK]:
        return Action.HEAVY_ATTACK
    if (keys[pygame.K_j] or keys[pygame.K_z]) and mask[Action.LIGHT_ATTACK]:
        return Action.LIGHT_ATTACK

    # 6. Jump
    if (keys[pygame.K_w] or keys[pygame.K_SPACE] or keys[pygame.K_UP]) and mask[Action.JUMP]:
        return Action.JUMP

    # 7. Movement
    if (keys[pygame.K_a] or keys[pygame.K_LEFT]) and mask[Action.MOVE_LEFT]:
        return Action.MOVE_LEFT
    if (keys[pygame.K_d] or keys[pygame.K_RIGHT]) and mask[Action.MOVE_RIGHT]:
        return Action.MOVE_RIGHT

    return Action.IDLE


def run_training_worker(weapon: str, mode: str, status_dict: dict, stop_event: threading.Event):
    """Background training worker to avoid blocking Pygame window."""
    status_dict["status"] = f"Initializing {weapon.capitalize()} training environment..."
    try:
        from train import make_env
        from sb3_contrib import MaskablePPO

        target_steps = 20_000 if mode == "single" else 60_000
        env = make_env(weapon=weapon, fight=1)

        model = MaskablePPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=4,
            verbose=0,
        )

        chunk = 2000
        steps = 0
        while steps < target_steps and not stop_event.is_set():
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            steps += chunk
            pct = int((steps / target_steps) * 100)
            status_dict["status"] = f"Training {weapon.capitalize()} — {pct}% ({steps:,}/{target_steps:,} steps)"

        if not stop_event.is_set():
            os.makedirs("models", exist_ok=True)
            save_path = os.path.join("models", f"menu_{weapon}_{mode}_final")
            model.save(save_path)
            status_dict["status"] = f"Complete! Saved to {save_path}.zip"
            status_dict["is_training"] = False
        else:
            status_dict["status"] = "Training stopped by user."
            status_dict["is_training"] = False
    except Exception as e:
        status_dict["status"] = f"Training notice: {str(e)[:50]}"
        status_dict["is_training"] = False


def main():
    parser = argparse.ArgumentParser(description="Boss Gauntlet: Interactive Play and Menus")
    parser.add_argument("--fight", type=int, default=1, help="Starting fight index (1..30)")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--upgrade", type=str, default="health", choices=["health", "flask"])
    parser.add_argument("--skip-menu", action="store_true", help="Start directly in combat mode")
    parser.add_argument("--mode", type=str, default="player", choices=["player", "agent"],
                        help="Initial combat mode: player or agent")
    parser.add_argument("--no-adapt", action="store_true",
                        help="Disable adaptive boss (boss does not learn from player)")
    args = parser.parse_args()

    renderer = Renderer()

    # ---- Adaptive boss PlayerModel (persists across fights / retries) ----
    _MODEL_JSON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "player_model.json")
    player_model = PlayerModel()
    if not args.no_adapt and os.path.exists(_MODEL_JSON_PATH):
        try:
            with open(_MODEL_JSON_PATH, "r", encoding="utf-8") as _f:
                player_model = PlayerModel.from_json(_f.read())
            print(f"[Adaptive Boss] Loaded player model ({player_model.total_samples} samples)")
        except Exception as _e:
            print(f"[Adaptive Boss] Could not load model: {_e} – starting fresh")
            player_model = PlayerModel()
    _adapt_enabled = not args.no_adapt

    def _save_player_model():
        """Persist the player model to JSON next to play.py."""
        if not _adapt_enabled:
            return
        try:
            with open(_MODEL_JSON_PATH, "w", encoding="utf-8") as _f:
                _f.write(player_model.to_json())
        except Exception as _e:
            print(f"[Adaptive Boss] Could not save model: {_e}")

    # State variables
    state = GameState.PLAYING if args.skip_menu else GameState.MAIN_MENU
    active_mode = args.mode          # "player" or "agent"
    selected_char = args.weapon      # "katana" or "revolver"
    current_fight = max(1, min(C.N_FIGHTS, args.fight))
    upgrade_path = args.upgrade

    # Menu selection indices
    main_menu_options = [
        "1. PLAYER VS BOSS",
        "2. AGENT VS BOSS",
        "3. TRAIN AGENTS",
        "4. QUIT GAME",
    ]
    selected_menu_idx = 0
    selected_char_btn = 0
    selected_train_btn = 0
    selected_game_over_btn = 0
    selected_victory_btn = 0

    # Training state
    train_weapon = "katana"
    train_mode = "single"
    training_status = {"status": "", "is_training": False}
    stop_training_event = threading.Event()
    train_thread = None

    # Gameplay objects
    env = None
    agent = None
    obs = None
    fight_start_time = 0.0
    boss_start_hp = 100.0
    stats = {}

    def init_battle(weapon_name: str, fight_idx: int):
        nonlocal env, agent, obs, fight_start_time, boss_start_hp
        env = BossFightEnv(weapon=weapon_name, fight=fight_idx, upgrade=upgrade_path, frame_skip=1)
        # Attach adaptive model only for human (player) mode
        if _adapt_enabled and active_mode == "player":
            player_model.set_fight(fight_idx)
            env.set_player_model(player_model)
        obs, _ = env.reset()
        fight_start_time = time.time()
        boss_start_hp = float(env.boss.hp)

        agent = None
        if active_mode == "agent":
            model_path = find_latest_model(weapon=weapon_name, fight=fight_idx)
            if model_path and os.path.exists(model_path) and MaskablePPO is not None:
                try:
                    print(f"Loading trained MaskablePPO model from {model_path}...")
                    agent = MaskablePPO.load(model_path)
                except Exception as e:
                    print(f"Failed to load {model_path}: {e}. Falling back to RuleBasedAgent.")
                    agent = RuleBasedAgent(weapon=weapon_name)
            else:
                print(f"Using RuleBasedAgent for {weapon_name} (Fight {fight_idx})...")
                agent = RuleBasedAgent(weapon=weapon_name)

    if state == GameState.PLAYING:
        init_battle(selected_char, current_fight)

    running = True

    while running:
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                running = False
                break

            # -------------------------------------------------------------
            # 1. MAIN MENU INPUT
            # -------------------------------------------------------------
            if state == GameState.MAIN_MENU:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected_menu_idx = (selected_menu_idx - 1) % len(main_menu_options)
                    elif event.key == pygame.K_DOWN:
                        selected_menu_idx = (selected_menu_idx + 1) % len(main_menu_options)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if selected_menu_idx == 0:
                            active_mode = "player"
                            state = GameState.CHAR_SELECT
                            selected_char_btn = 0
                        elif selected_menu_idx == 1:
                            active_mode = "agent"
                            state = GameState.CHAR_SELECT
                            selected_char_btn = 0
                        elif selected_menu_idx == 2:
                            state = GameState.TRAIN_MENU
                            selected_train_btn = 0
                        elif selected_menu_idx == 3:
                            running = False
                    elif event.key == pygame.K_1:
                        active_mode = "player"
                        state = GameState.CHAR_SELECT
                        selected_char_btn = 0
                    elif event.key == pygame.K_2:
                        active_mode = "agent"
                        state = GameState.CHAR_SELECT
                        selected_char_btn = 0
                    elif event.key == pygame.K_3:
                        state = GameState.TRAIN_MENU
                        selected_train_btn = 0
                    elif event.key in (pygame.K_4, pygame.K_ESCAPE):
                        running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, b_rect in enumerate(btn_rects):
                        if b_rect.collidepoint(event.pos):
                            if i == 0:
                                active_mode = "player"
                                state = GameState.CHAR_SELECT
                                selected_char_btn = 0
                            elif i == 1:
                                active_mode = "agent"
                                state = GameState.CHAR_SELECT
                                selected_char_btn = 0
                            elif i == 2:
                                state = GameState.TRAIN_MENU
                                selected_train_btn = 0
                            elif i == 3:
                                running = False

            # -------------------------------------------------------------
            # 2. CHARACTER SELECT INPUT
            # -------------------------------------------------------------
            elif state == GameState.CHAR_SELECT:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        if selected_char_btn == 0:
                            current_fight = max(1, current_fight - 1)
                        else:
                            selected_char = "katana"
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        if selected_char_btn == 0:
                            current_fight = min(C.N_FIGHTS, current_fight + 1)
                        else:
                            selected_char = "revolver"
                    elif event.key == pygame.K_TAB:
                        selected_char = "revolver" if selected_char == "katana" else "katana"
                    elif event.key in (pygame.K_UP, pygame.K_DOWN):
                        selected_char_btn = 1 - selected_char_btn
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if selected_char_btn == 0:
                            init_battle(selected_char, current_fight)
                            state = GameState.PLAYING
                        else:
                            state = GameState.MAIN_MENU
                    elif event.key == pygame.K_ESCAPE:
                        state = GameState.MAIN_MENU

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if "card_katana" in char_elements and char_elements["card_katana"].collidepoint(event.pos):
                        selected_char = "katana"
                    elif "card_revolver" in char_elements and char_elements["card_revolver"].collidepoint(event.pos):
                        selected_char = "revolver"
                    elif "arrow_left" in char_elements and char_elements["arrow_left"].collidepoint(event.pos):
                        current_fight = max(1, current_fight - 1)
                    elif "arrow_right" in char_elements and char_elements["arrow_right"].collidepoint(event.pos):
                        current_fight = min(C.N_FIGHTS, current_fight + 1)
                    elif "start" in char_elements and char_elements["start"].collidepoint(event.pos):
                        init_battle(selected_char, current_fight)
                        state = GameState.PLAYING
                    elif "back" in char_elements and char_elements["back"].collidepoint(event.pos):
                        state = GameState.MAIN_MENU

            # -------------------------------------------------------------
            # 3. TRAIN MENU INPUT
            # -------------------------------------------------------------
            elif state == GameState.TRAIN_MENU:
                if not training_status["is_training"]:
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_UP, pygame.K_DOWN):
                            selected_train_btn = 1 - selected_train_btn
                        elif event.key == pygame.K_TAB:
                            train_weapon = "revolver" if train_weapon == "katana" else "katana"
                        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                            if selected_train_btn == 0:
                                training_status["is_training"] = True
                                stop_training_event.clear()
                                train_thread = threading.Thread(
                                    target=run_training_worker,
                                    args=(train_weapon, train_mode, training_status, stop_training_event),
                                    daemon=True,
                                )
                                train_thread.start()
                            else:
                                state = GameState.MAIN_MENU
                        elif event.key == pygame.K_ESCAPE:
                            state = GameState.MAIN_MENU

                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if "weapon_katana" in train_elements and train_elements["weapon_katana"].collidepoint(event.pos):
                            train_weapon = "katana"
                        elif "weapon_revolver" in train_elements and train_elements["weapon_revolver"].collidepoint(event.pos):
                            train_weapon = "revolver"
                        elif "mode_single" in train_elements and train_elements["mode_single"].collidepoint(event.pos):
                            train_mode = "single"
                        elif "mode_curriculum" in train_elements and train_elements["mode_curriculum"].collidepoint(event.pos):
                            train_mode = "curriculum"
                        elif "start" in train_elements and train_elements["start"].collidepoint(event.pos):
                            training_status["is_training"] = True
                            stop_training_event.clear()
                            train_thread = threading.Thread(
                                target=run_training_worker,
                                args=(train_weapon, train_mode, training_status, stop_training_event),
                                daemon=True,
                            )
                            train_thread.start()
                        elif "back" in train_elements and train_elements["back"].collidepoint(event.pos):
                            state = GameState.MAIN_MENU
                else:
                    # In training progress view
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        stop_training_event.set()
                        training_status["status"] = "Stopping training..."
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        if "cancel" in train_elements and train_elements["cancel"].collidepoint(event.pos):
                            stop_training_event.set()
                            training_status["status"] = "Stopping training..."

            # -------------------------------------------------------------
            # 4. GAME OVER INPUT
            # -------------------------------------------------------------
            elif state == GameState.GAME_OVER:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        selected_game_over_btn = (selected_game_over_btn - 1) % 3
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected_game_over_btn = (selected_game_over_btn + 1) % 3
                    elif event.key in (pygame.K_r,):
                        # Retry fight
                        init_battle(selected_char, current_fight)
                        state = GameState.PLAYING
                    elif event.key in (pygame.K_m, pygame.K_ESCAPE):
                        state = GameState.MAIN_MENU
                    elif event.key in (pygame.K_q,):
                        running = False
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if selected_game_over_btn == 0:
                            init_battle(selected_char, current_fight)
                            state = GameState.PLAYING
                        elif selected_game_over_btn == 1:
                            state = GameState.MAIN_MENU
                        elif selected_game_over_btn == 2:
                            running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, b_rect in enumerate(game_over_btns):
                        if b_rect.collidepoint(event.pos):
                            if i == 0:
                                init_battle(selected_char, current_fight)
                                state = GameState.PLAYING
                            elif i == 1:
                                state = GameState.MAIN_MENU
                            elif i == 2:
                                running = False

            # -------------------------------------------------------------
            # 5. VICTORY INPUT
            # -------------------------------------------------------------
            elif state == GameState.VICTORY:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        selected_victory_btn = (selected_victory_btn - 1) % 4
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        selected_victory_btn = (selected_victory_btn + 1) % 4
                    elif event.key in (pygame.K_SPACE,):
                        # Next Fight
                        if current_fight < C.N_FIGHTS:
                            current_fight += 1
                        init_battle(selected_char, current_fight)
                        state = GameState.PLAYING
                    elif event.key in (pygame.K_r,):
                        # Retry Fight
                        init_battle(selected_char, current_fight)
                        state = GameState.PLAYING
                    elif event.key in (pygame.K_m, pygame.K_ESCAPE):
                        state = GameState.MAIN_MENU
                    elif event.key in (pygame.K_q,):
                        running = False
                    elif event.key in (pygame.K_RETURN,):
                        if selected_victory_btn == 0:
                            if current_fight < C.N_FIGHTS:
                                current_fight += 1
                            init_battle(selected_char, current_fight)
                            state = GameState.PLAYING
                        elif selected_victory_btn == 1:
                            init_battle(selected_char, current_fight)
                            state = GameState.PLAYING
                        elif selected_victory_btn == 2:
                            state = GameState.MAIN_MENU
                        elif selected_victory_btn == 3:
                            running = False

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for i, b_rect in enumerate(victory_btns):
                        if b_rect.collidepoint(event.pos):
                            if i == 0:
                                if current_fight < C.N_FIGHTS:
                                    current_fight += 1
                                init_battle(selected_char, current_fight)
                                state = GameState.PLAYING
                            elif i == 1:
                                init_battle(selected_char, current_fight)
                                state = GameState.PLAYING
                            elif i == 2:
                                state = GameState.MAIN_MENU
                            elif i == 3:
                                running = False

            # -------------------------------------------------------------
            # 6. IN-GAME COMBAT INPUT
            # -------------------------------------------------------------
            elif state == GameState.PLAYING:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        state = GameState.MAIN_MENU
                    elif event.key == pygame.K_r:
                        init_battle(selected_char, current_fight)
                    elif pygame.K_1 <= event.key <= pygame.K_9:
                        current_fight = event.key - pygame.K_0
                        init_battle(selected_char, current_fight)

        # -------------------------------------------------------------
        # STATE UPDATE & RENDERING
        # -------------------------------------------------------------
        if state == GameState.MAIN_MENU:
            btn_rects = renderer.draw_main_menu(
                options=main_menu_options,
                selected_idx=selected_menu_idx,
                mouse_pos=mouse_pos,
            )

        elif state == GameState.CHAR_SELECT:
            char_elements = renderer.draw_character_select(
                mode_name="Player vs Boss" if active_mode == "player" else "Agent vs Boss",
                selected_char=selected_char,
                starting_fight=current_fight,
                selected_btn=selected_char_btn,
                mouse_pos=mouse_pos,
            )

        elif state == GameState.TRAIN_MENU:
            train_elements = renderer.draw_training_menu(
                selected_weapon=train_weapon,
                selected_mode=train_mode,
                selected_btn=selected_train_btn,
                is_training=training_status["is_training"],
                training_status=training_status["status"],
                mouse_pos=mouse_pos,
            )

        elif state == GameState.PLAYING:
            keys = pygame.key.get_pressed()

            # Choose action based on mode
            if active_mode == "player":
                action = get_action_from_keys(keys, env)
            else:
                # Agent action
                if agent is not None:
                    try:
                        action_val, _ = agent.predict(obs, action_masks=env.action_masks())
                        action = Action(int(action_val))
                    except Exception:
                        action = Action.IDLE
                else:
                    action = Action.IDLE

            obs, reward, term, trunc, info = env.step(action)
            renderer.draw(env)

            # Check outcome
            if term or trunc:
                combat_duration = max(0.1, time.time() - fight_start_time)
                damage_dealt = max(0.0, boss_start_hp - env.boss.hp)
                stats = {
                    "damage": damage_dealt,
                    "duration": combat_duration,
                    "boss_hp": max(0.0, env.boss.hp),
                    "max_hp": env.boss.max_hp,
                    "hearts": max(0, env.player.hearts),
                    "max_hearts": env.player.max_hearts,
                }

                # Persist adaptive model after each fight
                _save_player_model()

                if info.get("win", False):
                    state = GameState.VICTORY
                    selected_victory_btn = 0
                else:
                    state = GameState.GAME_OVER
                    selected_game_over_btn = 0

        elif state == GameState.GAME_OVER:
            game_over_btns = renderer.draw_game_over_screen(
                fight=current_fight,
                stats=stats,
                selected_btn=selected_game_over_btn,
                mouse_pos=mouse_pos,
            )

        elif state == GameState.VICTORY:
            victory_btns = renderer.draw_victory_screen(
                fight=current_fight,
                stats=stats,
                selected_btn=selected_victory_btn,
                mouse_pos=mouse_pos,
            )

    _save_player_model()
    if _adapt_enabled:
        print(f"[Adaptive Boss] Model saved ({player_model.total_samples} samples, "
              f"alpha={player_model.alpha:.2f})")
        tend = player_model.tendencies()
        top = sorted(tend.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print("[Adaptive Boss] Your top tendencies:")
        for r, rate in top:
            print(f"  {r:14s}: {rate*100:.0f}%")

    renderer.close()


if __name__ == "__main__":
    main()
