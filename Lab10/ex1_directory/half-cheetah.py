import gymnasium as gym
import mujoco_py
env = gym.make('HalfCheetah-v4', render_mode='human')

observation, info = env.reset(seed=42)

for _ in range(600):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)
env.close()