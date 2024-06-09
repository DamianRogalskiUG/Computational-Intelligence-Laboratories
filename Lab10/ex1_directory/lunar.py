
import gymnasium as gym

env = gym.make('LunarLander-v2', render_mode='human')
observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = env.action_space.sample()  # Losowa akcja
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()
env.close()

