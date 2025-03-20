import math
import numpy as np

def function(z: list):
    x, y = z
    return 9*x*y / math.e**(x**2 + 1/2*x + y**2)
def dfdx(z: list):
    x, y = z
    return (-18*x**2 - 4.5*x + 9)*y / math.e**(x**2 + x/2 + y**2)
def dfdy(z: list):
    x, y = z
    return -9*x * (2*y**2 - 1) / math.e**(x**2 + x/2 + y**2)
def gradient(z: list):
    return np.array([dfdx(z), dfdy(z)])

def sgd_minimum(initial: list, alpha=0.1, steps=1000):
    x_n = list(initial)
    grad = gradient(x_n)
    diff = -alpha * grad
    for _ in range(steps):
        diff = -alpha * gradient(x_n)
        x_n += diff
        if np.all(np.abs(diff) <= 1e-06):
            break
    return x_n

def sgd_maximum(initial: list, alpha=0.1, steps=1000):
    x_n = list(initial)
    grad = np.array(gradient(x_n))
    diff = alpha * grad
    for _ in range(steps):
        diff = alpha * gradient(x_n)
        x_n += diff
        if np.all(np.abs(diff) <= 1e-06):
            break
    return x_n

print("Punkt początkowy: [0.6, 0.71]")
print(sgd_minimum([0.6, 0.71]))
# print("Punkt początkowy: [1, 0.71]")
# print(sgd_minimum([1, 0.71]))
# print("Punkt początkowy: [-0.9, -0.8]")
# print(sgd_minimum([-0.9, -0.8]))
# print("Punkt początkowy: [-0.85, -0.79]")
# print(sgd_minimum([-0.85, -0.79]))
# print("Punkt początkowy: [-2, 2]")
# print(sgd_minimum([-2, 2]))
# print("Punkt początkowy: [1, -0.71]")
# print(sgd_minimum([1, -0.71]))

# print("Punkt początkowy: [1, -2]")
# print(sgd_maximum([1, -2]))
# print("Punkt początkowy: [2, -2]")
# print(sgd_maximum([2, -2]))
# print("Punkt początkowy: [-2, 1]")
# print(sgd_maximum([-2, 1]))
# print("Punkt początkowy: [-1, 1]")
# print(sgd_maximum([-1, 1]))
# print("Punkt początkowy: [0, 1]")
# print(sgd_maximum([0, 1]))
# print("Punkt początkowy: [-1, 0]")
# print(sgd_maximum([-1, 0]))
# print(sgd_maximum([-5, -5]))