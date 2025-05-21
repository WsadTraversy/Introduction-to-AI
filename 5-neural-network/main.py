import copy
import os
import pickle
import pygame
import time
import torch

from food import Food
from MLP import MLP
from read_data import game_state_to_data_sample, game_state_to_data_sample_tail
from snake import Snake, Direction


def main():
    pygame.init()
    bounds = (300, 300)
    window = pygame.display.set_mode(bounds)
    pygame.display.set_caption("Snake")

    block_size = 30
    snake = Snake(block_size, bounds)
    food = Food(block_size, bounds, lifetime=100)

    agent = BehavioralCloningAgent(block_size, bounds)  # Once your agent is good to go, change this line
    scores = []
    run = True
    pygame.time.delay(1000)
    while run:
        pygame.time.delay(200)  # Adjust game speed, decrease to test your agent and model quickly

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        game_state = {"food": (food.x, food.y),
                      "snake_body": snake.body,  # The last element is snake's head
                      "snake_direction": snake.direction}
        
        direction = agent.act(game_state)
        snake.turn(direction)

        snake.move()
        snake.check_for_food(food)
        food.update()

        if snake.is_wall_collision() or snake.is_tail_collision():
            pygame.display.update()
            pygame.time.delay(300)
            scores.append(snake.length - 3)
            snake.respawn()
            food.respawn()

        window.fill((0, 0, 0))
        snake.draw(pygame, window)
        food.draw(pygame, window)
        pygame.display.update()

    print(f"Scores: {scores}")
    agent.dump_data()
    pygame.quit()


class HumanAgent:
    """ In every timestep every agent should perform an action (return direction) based on the game state. Please note, that
    human agent should be the only one using the keyboard and dumping data. """
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)  # Last 10 frames are when you press exit, so they are bad, skip them


class BehavioralCloningAgent:
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.model = MLP(1, 4)
        self.model.load_state_dict(torch.load('data/state_dict.pickle'))

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        action = Direction.DOWN
        processed_state = []
        processed_state.append(game_state)
        data_sample = game_state_to_data_sample(processed_state)
        self.model.eval()
        y_pred = self.model(data_sample)
        indices = torch.argsort(y_pred, descending=True)
        val = int(indices[0][0])
        second_val = int(indices[0][1])

        current_direction = game_state['snake_direction']

        if val == 0 and current_direction != Direction.DOWN:
            action = self.get_direction(val)
        elif val == 0:
            action = self.get_direction(second_val)
        if val == 1 and current_direction != Direction.LEFT:
            action = self.get_direction(val)
        elif val == 1:
            action = self.get_direction(second_val)  
        if val == 2 and current_direction != Direction.UP:
            action = self.get_direction(val)
        elif val == 2:
            action = self.get_direction(second_val)
        if val == 3 and current_direction != Direction.RIGHT:
            action = self.get_direction(val)
        elif val == 3:
            action = self.get_direction(second_val)

        return action
    
    def get_direction(self, num):
        if num == 0:
            return Direction.UP
        elif num == 1:
            return Direction.RIGHT
        elif num == 2:
            return Direction.DOWN
        elif num == 3:
            return Direction.LEFT

    def dump_data(self):
        pass


if __name__ == "__main__":
    main()