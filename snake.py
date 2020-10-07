"""
Created by Edward Li at 10/6/20
"""

import pygame
import numpy as np
import random
from collections import deque


def update_screen():
    pygame.display.update()


class Snake:
    def __init__(self, width, height, speed = 20 ,margin = 20):
        pygame.display.set_caption("Snake_RL")
        self.width = width
        self.height = height
        self.bg = pygame.image.load("assets/background.png")
        self.player = Player(self, speed)
        self.food = Food(self)
        self.score = 0
        self.crash = False
        self.margin = 20


class Player:
    def __init__(self,game,speed = 20):
        # start in the center of the screen
        self.x = 0.5 * game.game_width
        self.y = 0.5 * game.game_height
        self.x = self.x - self.x % 20
        self.y = self.y - self.y % 20
        self.speed = speed
        self.position = deque()
        self.position_dict = {}
        self.position.append([self.x, self.y])
        self.image = pygame.image.load("assets/green_square.jpg")
        self.delta_x = speed
        self.delta_y = 0

        self.eaten = False

    def update_position(self,x,y):
        if x != self.position[-1][0] and y != self.position[-1][1]:
            if self.food >1:
                xp,yp = self.position.popleft()
                del self.position_dict[(xp,yp)]
            self.position.append([x,y])
            self.position_dict[(x,y)] = 1

    def move(self,move,x,y,game,food):
        """
        handle a move from the agent/player.
        :param move:
        :param x:
        :param y:
        :param game:
        :param food:
        :return:
        """
        if self.eaten:
            self.position.append([x,y])
            self.eaten = False
            self.food += 1
        # check if we moved left or right. or no move.
        if np.array_equal(move, [0, 1, 0]) and self.delta_y == 0:
            # right - going horizontal
            self.delta_y = self.delta_x
            self.delta_x = 0
        elif np.array_equal(move, [0, 1, 0]) and self.delta_x == 0:
            # right - going vertical
            self.delta_x = -self.delta_y
            self.delta_y = 0
        elif np.array_equal(move, [0, 0, 1]) and self.delta_y == 0:
            # left - going horizontal
            self.delta_x = 0
            self.delta_y = -self.delta_x
        elif np.array_equal(move, [0, 0, 1]) and self.delta_x == 0:
            # left - going vertical
            self.delta_x = self.delta_y,
            self.delta_y = 0
        self.x = x + self.delta_x
        self.y = x + self.delta_y

        if self.x < game.margin or self.x > game.game_width - game.margin \
                or self.y < game.margin \
                or self.y > game.game_height - game.margin \
                or (self.x, self.y) in self.position:
            game.crash = True

        if self.x == food.x_food and self.y == food.y_food:
            food.next_food(game, self)
            self.eaten = True
            game.score = game.score + 1

        self.update_position(self.x,self.y)

    def display_player(self,game):
        if not game.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food:
    def __init__(self,game):
        self.image = pygame.image.load("assets/apple.png")
        self.x_food = random.randint(game.margin,game.width-game.margin)
        self.y_food = random.randint(game.margin,game.height-game.margin)

    def next_food(self,game,player):
        self.x_food = random.randint(game.margin, game.width - game.margin)
        self.x_food = self.x_food - self.x_food % game.speed

        self.y_food = random.randint(game.margin, game.height - game.margin)
        self.y_food = self.y_food - self.y_food % game.speed
        if (self.x_food, self.y_food) not in player.position_dict:
            return self.x_food, self.y_food
        else:
            self.next_food(game, player)

    def display_food(self,game):
        game.gameDisplay.blit(self.image,(self.x_food,self.y_food))
        update_screen()