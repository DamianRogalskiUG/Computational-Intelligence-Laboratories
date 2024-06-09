import gymnasium as gym
import numpy as np

def policy(observation):
    row, col = divmod(observation, 8)
    if col < 7:
        return 2
    elif row < 7:
        return 1
    else:
        return 0

env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human")

observation, info = env.reset(seed=42)

for _ in range(30):
    action = policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()
