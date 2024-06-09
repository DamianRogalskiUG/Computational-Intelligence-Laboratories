import gym

# env = gym.make('CartPole-v1', render_mode='human')
#
# observation, info = env.reset(seed=42)

env = gym.make('CartPole-v1', render_mode='human')
env.reset()
for _ in range(200):
    env.render()
    # Prosta polityka: Move right if pole angle is positive, else move left
    action = 1 if env.unwrapped.state[2] > 0 else 0
    env.step(action)
env.close()