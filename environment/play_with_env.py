from env import GridWorldEnv 
import pygame

if __name__ == '__main__':
    env = GridWorldEnv(render_mode="ansi")
    obs, _ = env.reset()
    env.render()
    # terminated = False
    # truncated = False

    # running = True
    # while running:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             running = False

    #     if not terminated and not truncated:
    #         action = env.action_space.sample()  # Random action
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         print(f"Reward: {reward}, Terminated: {terminated}")
    #     else:
    #         obs, _ = env.reset()  # Reset if episode is over
    #         terminated = False
    #         truncated = False

    # env.close()