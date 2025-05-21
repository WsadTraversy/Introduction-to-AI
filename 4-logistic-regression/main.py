import copy
import os
import pickle
import pygame
import time
import operator

from food import Food
from model import game_state_to_data_sample, teached_models
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
        clf_up, clf_right, clf_down, clf_left = teached_models()
        self.clf_up = clf_up
        self.clf_right = clf_right
        self.clf_down = clf_down
        self.clf_left = clf_left
        # self.model = teached_models()

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        action = Direction.DOWN
        processed_state = []
        processed_state.append(game_state)
        data_sample = game_state_to_data_sample(processed_state)
        
        predicted_values = {"up":self.clf_up.predict(data_sample),
                           "right":self.clf_right.predict(data_sample),
                           "down":self.clf_down.predict(data_sample),
                           "left":self.clf_left.predict(data_sample)}
        predicted_values = sorted(predicted_values.items(), key=operator.itemgetter(1))
        val = predicted_values[-1][0]
        second_val = predicted_values[-2][0]

        current_direction = game_state['snake_direction']

        if val == 'up' and current_direction != Direction.DOWN:
            action = self.get_direction(val)
        elif val == 'up':
            action = self.get_direction(second_val)
        if val == 'right' and current_direction != Direction.LEFT:
            action = self.get_direction(val)
        elif val == 'right':
            action = self.get_direction(second_val)  
        if val == 'down' and current_direction != Direction.UP:
            action = self.get_direction(val)
        elif val == 'down':
            action = self.get_direction(second_val)
        if val == 'left' and current_direction != Direction.RIGHT:
            action = self.get_direction(val)
        elif val == 'left':
            action = self.get_direction(second_val)

        return action
        
    def get_direction(self, num):
        if num == 'up':
            return Direction.UP
        elif num == 'right':
            return Direction.RIGHT
        elif num == 'down':
            return Direction.DOWN
        elif num == 'left':
            return Direction.LEFT
    
    def dump_data(self):
        pass


if __name__ == "__main__":
    main()