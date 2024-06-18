import gymnasium as gym
import numpy as np


def heuristic_car_policy(observation):
    height, width, _ = observation.shape
    middle_line = observation[height // 2:height // 2 + 1, :, 0]
    left = np.mean(middle_line[:, :width // 3])
    center = np.mean(middle_line[:, width // 3: 2 * width // 3])
    right = np.mean(middle_line[:, 2 * width // 3:])

    steering = 0
    if center < left and center < right:
        steering = 0
    elif left < right:
        steering = -1
    else:
        steering = 1

    acceleration = 0.5
    brake = 0

    return np.array([steering, acceleration, brake])


env = gym.make('CarRacing-v2', render_mode='human')
observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = heuristic_car_policy(observation)
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()
env.close()

# discrete actions,