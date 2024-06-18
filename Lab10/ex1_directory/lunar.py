import gymnasium as gym
import numpy as np

def heuristic_lander_policy(state):
    angle_targ = state[0]*0.5 + state[2]*1.0
    if angle_targ > 0.4: angle_targ = 0.4
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(state[0])

    angle_todo = (angle_targ - state[4])*0.5 - (state[5])*1.0
    hover_todo = (hover_targ - state[1])*0.5 - (state[3])*0.5

    if state[6] or state[7]:
        angle_todo = 0
        hover_todo = -(state[3])*0.5

    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        a = 2
    elif hover_todo < -np.abs(angle_todo) and hover_todo < -0.05:
        a = 1
    elif angle_todo < -0.05:
        a = 3
    elif angle_todo > 0.05:
        a = 2
    return a

env = gym.make('LunarLander-v2', render_mode='human')
observation, info = env.reset()
for _ in range(1000):
    env.render()
    action = heuristic_lander_policy(observation)  # Akcja na podstawie polityki heurystycznej
    observation, reward, done, truncated, info = env.step(action)
    if done:
        observation, info = env.reset()
env.close()
