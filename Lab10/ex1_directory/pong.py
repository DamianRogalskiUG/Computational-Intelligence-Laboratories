import gymnasium as gym

env = gym.make('Pong-v0', render_mode='rgb_array')

observation, info = env.reset(seed=42)

for _ in range(600):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset(seed=42)
env.close()