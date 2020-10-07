"""
Created by Edward Li at 10/6/20
"""

import pygame
import numpy as np
from collections import deque

class Snake:
    def __init__(self, width, height, speed):
        pygame.display.set_caption("Snake_RL")
        self.width = width
        self.height = height
        self.bg = pygame.image.load("assets/background.png")
        self.player = Player(self, speed)
        self.food = Food()
        self.score = 0
        self.crash = False


class Player:
    def __init__(self,game,speed = 20):
        # start in the center of the screen
        self.x = 0.5 * game.game_width
        self.y = 0.5 * game.game_height
        self.x = self.x - self.x % 20
        self.y = self.y - self.y % 20
        self.speed = speed
        self.position = deque()
        self.position.append([self.x, self.y])
        self.image = pygame.image.load("assets/green_square.jpg")
        self.delta_x = speed
        self.delta_y = 0

    def update_position(self,x,y):
        if x != self.position[-1][0] and y != self.position[-1][1]:
            if self.food


class Food:
    def __init__(self):
        self.image = pygame.image.load("assets/snake.png")
