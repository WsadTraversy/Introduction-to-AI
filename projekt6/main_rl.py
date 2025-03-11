import random
import pickle
import matplotlib.pyplot as plt
import pygame
import torch
import os
import time
import copy

from food import Food
from snake import Snake, Direction


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds)

    agent = QLearningAgent(block_size, bounds, 0.2, 0.99, is_training=False)
    scores = []
    rewards_box = []
    rewards = 0
    run = True
    pygame.time.delay(1000)
    reward, is_terminal = 0, False
    episode, total_episodes = 0, 50
    while episode < total_episodes and run:
        pygame.time.delay(1)  # Adjust game speed, decrease to learn agent faster

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}

        direction = agent.act(game_state, reward, is_terminal)
        reward = -0.001
        is_terminal = False
        snake.turn(direction)
        snake.move()
        reward += snake.check_for_food(food)
        rewards += reward
        food.update()
        
        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(1)
            scores.append(snake.length - 3)
            rewards_box.append(rewards)
            snake.respawn()
            food.respawn()
            episode += 1
            reward -= 0.999
            is_terminal = True

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()
        
    print(f"Scores: {scores}")
    # This will create a smoothed mean score per episode plot.
    # I want you to create a smoothed sum of rewards per episode plots, that's how we evaluate RL algorithms!
    scores = torch.tensor(scores, dtype=torch.float).unsqueeze(0)
    scores = torch.nn.functional.avg_pool1d(scores, 31, stride=1)
    plt.plot(scores.squeeze(0))
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.savefig("mean_score.png")

    rewards_box = torch.tensor(rewards_box, dtype=torch.float).unsqueeze(0)
    plt.plot(rewards_box.squeeze(0))
    plt.xlabel("episode")
    plt.ylabel("sum of rewards")
    plt.savefig("sum_of_reward.png")
    
    agent.dump_qfunction()
    pygame.quit()


class QLearningAgent:
    def __init__(self, block_size, bounds, discount=0.9, epsilon=0.9, lr=0.01, is_training=False, load_qfunction_path=None):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.is_training = is_training
        self.Q = torch.zeros((2,2,2,2,2,2,2,2,2,2,2,4,4))
        self.obs = None
        self.action = None
        self.discount = discount
        self.epsilon = epsilon
        self.lr = lr
        self.data = []

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        new_obs = self.game_state_to_observation(game_state)
        new_action = random.randint(0, 3)
        if random.random() < self.epsilon:
            new_action = torch.argmax(self.Q[new_obs])
        if self.action is not None:
            if not is_terminal:
                update = reward + self.discount * torch.max(self.Q[new_obs]) - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]
            self.Q[self.obs][self.action] += self.lr * update

        self.action = new_action
        self.obs = new_obs
        
        return Direction(int(new_action))

    @staticmethod
    def game_state_to_observation(game_state):
        gs = game_state
        food_up = int(gs["food"][1] < gs["snake_body"][-1][1])
        food_right = int(gs["food"][0] > gs["snake_body"][-1][0])
        food_down = int(gs["food"][1] > gs["snake_body"][-1][1])
        food_left = int(gs["food"][0] < gs["snake_body"][-1][0])
        collision_up = int(gs["snake_body"][-1][1] == 0)
        collision_right = int(gs["snake_body"][-1][0] == 270)
        collision_down = int(gs["snake_body"][-1][1] == 270)
        collision_left = int(gs["snake_body"][-1][0] == 0)
        tail_forward = 0
        tail_right  = 0
        tail_left = 0 
        body_values = list()
        head = gs["snake_body"][-1]
        for element in gs["snake_body"][:-1]:
            body_values.append((element[0], element[1]))
        state1 = head[0] + 30
        state2 = head[0] - 30
        state3 = head[1] + 30
        state4 = head[1] - 30

        if gs["snake_direction"].value == 0: # up
            if (head[0], state4) in body_values:
                tail_forward = 1
            if (state1, head[1]) in body_values:
                tail_right = 1
            if (state2, head[1]) in body_values:
                tail_left = 1
        
        if gs["snake_direction"].value == 1: # right
            if (state1, head[1]) in body_values:
                tail_forward = 1
            if (head[0], state3) in body_values:
                tail_right = 1
            if (head[0], state4) in body_values:
                tail_left = 1

        if gs["snake_direction"].value == 2: # down
            if (head[0], state3) in body_values:
                tail_forward = 1
            if (state1, head[1]) in body_values:
                tail_left = 1
            if (state2, head[1]) in body_values:
                tail_right = 1

        if gs["snake_direction"].value == 3: # left
            if (state2, head[1]) in body_values:
                tail_forward = 1
            if (head[0], state3) in body_values:
                tail_left = 1
            if (head[0], state4) in body_values:
                tail_right = 1
            
        return food_up, food_right, food_down, food_left, collision_up, collision_right, collision_down, collision_left, tail_right, tail_left, tail_forward, gs["snake_direction"].value

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        with open("data/Q-Learning-function.pickle", 'rb') as f:
            self.Q = pickle.load(f)
        new_obs = self.game_state_to_observation(game_state)
        new_action = torch.argmax(self.Q[new_obs])
        self.data.append((copy.deepcopy(game_state), Direction(int(new_action))))
        return Direction(int(new_action))

    def dump_qfunction(self):
        with open("data/Q-Learning-function.pickle", 'wb') as f:
            pickle.dump(self.Q, f)

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)  # Last 10 frames are when you press exit, so they are bad, skip them


if __name__ == "__main__":
    main()
