"""Interactive mode: Play the Boss Gauntlet with keyboard in a Pygame window.

Controls:
    A / Left Arrow      : Move Left
    D / Right Arrow     : Move Right
    W / Space / Up Arrow: Jump
    J / Z               : Light Attack
    K / X               : Heavy Attack
    L / C / LShift      : Dodge Roll
    U / F               : Parry
    I / Q               : Super (Blood Katana)
    H / E               : Drink Flask
    R                   : Restart Episode
    1 .. 9              : Jump to Fight N
    ESC                 : Quit
"""

import argparse
import sys
import pygame

from boss_env import constants as C
from boss_env.env import BossFightEnv, Action
from boss_env.renderer import Renderer


def get_action_from_keys(keys, env: BossFightEnv) -> Action:
    mask = env.action_masks()

    # Priority order for instantaneous actions:
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


def main():
    parser = argparse.ArgumentParser(description="Play Boss Gauntlet interactively")
    parser.add_argument("--fight", type=int, default=1, help="Starting fight index (1..30)")
    parser.add_argument("--weapon", type=str, default="katana", choices=["katana", "revolver"])
    parser.add_argument("--upgrade", type=str, default="health", choices=["health", "flask"])
    parser.add_argument("--lose-mode", type=str, default="fight", choices=["fight", "gauntlet"],
                        help="On loss: retry current fight or reset to fight 1")
    args = parser.parse_args()

    current_fight = args.fight
    env = BossFightEnv(weapon=args.weapon, fight=current_fight, upgrade=args.upgrade, frame_skip=1)
    renderer = Renderer()

    obs, _ = env.reset()
    running = True
    game_over = False
    game_result_text = ""

    print("\n" + "=" * 60)
    print(f"BOSS GAUNTLET - Playing Fight {current_fight}")
    print("Controls: A/D Move | W/Space Jump | J/K Attack | L Roll | U Parry | I Super | R Restart")
    print("=" * 60 + "\n")

    while running:
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Manual reset
                    obs, _ = env.reset()
                    game_over = False
                    game_result_text = ""
                # Quick jump to fight 1..9
                elif pygame.K_1 <= event.key <= pygame.K_9:
                    current_fight = event.key - pygame.K_0
                    env.set_fight(current_fight)
                    obs, _ = env.reset()
                    game_over = False
                    game_result_text = ""
                    print(f"Switched to Fight {current_fight}")

        keys = pygame.key.get_pressed()

        if not game_over:
            action = get_action_from_keys(keys, env)
            obs, reward, term, trunc, info = env.step(action)

            if term or trunc:
                game_over = True
                if info.get("win", False):
                    game_result_text = f"VICTORY! Fight {current_fight} Cleared!"
                    print(f"\n{game_result_text}")
                else:
                    reason = "Timeout" if trunc else "Defeated"
                    game_result_text = f"DEFEAT ({reason})"
                    print(f"\n{game_result_text}")
        else:
            # Game over state: press Space or R to continue
            if keys[pygame.K_SPACE] or keys[pygame.K_r]:
                if "VICTORY" in game_result_text:
                    # Advance to next fight
                    if current_fight < C.N_FIGHTS:
                        current_fight += 1
                        env.set_fight(current_fight)
                else:
                    if args.lose_mode == "gauntlet":
                        current_fight = 1
                        env.set_fight(1)

                obs, _ = env.reset()
                game_over = False
                game_result_text = ""

        # Draw frame
        renderer.draw(env)

        # Draw Game Over Banner overlay if applicable
        if game_over:
            overlay = pygame.Surface((renderer.width, renderer.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 140))
            renderer.screen.blit(overlay, (0, 0))

            color = (80, 240, 120) if "VICTORY" in game_result_text else (255, 60, 60)
            txt_res = renderer.font_title.render(game_result_text, True, color)
            renderer.screen.blit(txt_res, ((renderer.width - txt_res.get_width()) // 2, 300))

            prompt = "Press SPACE or R to continue / retry"
            txt_prompt = renderer.font_main.render(prompt, True, (220, 220, 220))
            renderer.screen.blit(txt_prompt, ((renderer.width - txt_prompt.get_width()) // 2, 360))

            pygame.display.flip()

    renderer.close()


if __name__ == "__main__":
    main()
