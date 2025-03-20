import math
import numpy as np
import random

def function(z: list):
    x, y = z
    return 9*x*y / math.e**(x**2 + 1/2*x + y**2)

# check if a point is within the bounds search
def in_bounds(point, bounds):
    for d in range(len(bounds)):
        if point[d] < bounds[d, 0] or point[d] > bounds[d, 1]:
            return False
    return True

def generate_population(bounds, size):
    population = list()
    for _ in range(size):
        candidate = None
        while candidate is None or not in_bounds(candidate, bounds):
            candidate = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, -1] - bounds[:, 0])
        population.append(candidate)
    return population

def mu_plus_lambda(mu, lam, bounds, sigma=0.5, steps=1000):
    best, best_eval = None, 1e+10
    # initial population
    population = generate_population(bounds, lam+mu)
    for epoch in range(steps):
        children = list()
        # crossing with mutation
        for _ in range(lam):
            gaussian_noise = np.random.normal(0, sigma ,size=2)
            a = np.random.randint(0, 2)
            i = np.random.randint(0, lam)
            j = np.random.randint(0, lam)
            child = a * population[i] + (1-a) * population[j] + gaussian_noise
            children.append(child)
        # evaluate fitness for the population
        scores = [function(c) for c in population]
        # rank scores in ascending order
        ranks = np.argsort(np.argsort(scores))
        # select the indexes for the top mu ranked solutions
        selected = [i for i,_ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
            # keep the parent
            children.append(population[i])
        # replace population with children
        population = children
    return best, best_eval, len(population)

bounds = np.asarray([[-10.0, -10.0], [-10.0, -10.0]])
best, best_eval, population_len = mu_plus_lambda(128, 512, bounds)
print(best)
print(best_eval)