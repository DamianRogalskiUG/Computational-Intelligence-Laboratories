import math
import random
import matplotlib.pyplot as plt
import numpy as np

v0 = 50
h = 100
g = 9.81
margin_of_error = 5

def calculate_distance(alpha):
    alpha_rad = math.radians(alpha)
    distance = (v0 * math.sin(alpha_rad) + math.sqrt(v0**2 * math.sin(alpha_rad)**2 + 2 * g * h))  * ( v0 * math.cos(alpha_rad)) / g
    return distance


def projectile_trajectory(alpha, x):
    alpha_rad = np.radians(alpha)
    y = -0.5 * g * (x / (v0 * np.cos(alpha_rad)))**2 + np.tan(alpha_rad) * x + h
    return y

target_distance = random.randint(50, 340)
print(f"Your target is {target_distance} meters away")

attempts = 0

running = True
while running:
    alpha = float(input(f"Input your alpha angle: "))

    distance = calculate_distance(alpha)

    if abs(distance - target_distance) <= margin_of_error:
        print("Cel trafiony!")
        print(f"Number of attempts: {attempts}")
        x_values = np.linspace(0, 340, 340)
        y_values = projectile_trajectory(alpha, x_values)

        plt.plot(x_values, y_values, color='blue')
        plt.grid(True)
        plt.xlabel('Dystans (m)')
        plt.ylabel('Wysokość (m)')
        plt.title('Trajektoria pocisku Warwolf')
        plt.savefig('trajektoria.png')
        plt.show()
        running = False
    else:
        print(f"Your shot missed the target (distance of the shot was {distance})")
        attempts += 1

# rozwiązania z chatuGPT były mocno przekombinowane pod względem wzorów i nie potrafiły narysować prawidłowej trajektorii