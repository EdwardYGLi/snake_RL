"""
Created by Edward Li at 10/6/20
"""

import argparse
import random
import sys
from collections import deque

import numpy as np
import pygame


def update_screen():
    pygame.display.update()


def display(player, food, game, record):
    game.gameDisplay.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(game)
    food.display_food(game)


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.gameDisplay.blit(text_score, (45, game.height + 20))
    game.gameDisplay.blit(text_score_number, (120, game.height + 20))
    game.gameDisplay.blit(text_highest, (190, game.height + 20))
    game.gameDisplay.blit(text_highest_number, (350, game.height + 20))
    game.gameDisplay.blit(game.bg, (0,0))


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


class Snake:
    def __init__(self, width, height, block_size=20, margin=20):
        pygame.display.set_caption("Snake_RL")
        self.width = width
        self.height = height
        self.bg = pygame.image.load("assets/background.png")
        self.bg = pygame.transform.scale(self.bg, (width, height))
        self.gameDisplay = pygame.display.set_mode((width, height + 40))
        self.score = 0
        self.crash = False
        self.margin = margin
        self.block_size = block_size
        self.player = Player(self)
        self.food = Food(self)
        
    def reset(self):
        self.player.reset()
        self.food.reset()
        self.score = 0
        self.crash = False


class Player:
    def __init__(self, game):
        self.game = game
        self.image = pygame.image.load("assets/green_square.jpg")
        self.image = pygame.transform.scale(self.image, (game.block_size, game.block_size))
        self.head_image = pygame.image.load("assets/snake_head.png")
        self.head_image = pygame.transform.scale(self.head_image, (game.block_size, game.block_size))
        # start in the center of the screen
        self.x = self.game.width // 2
        self.y = self.game.height // 2
        # mod by grid size so its grid aligned.
        self.x = self.x - self.x % self.game.block_size
        self.y = self.y - self.y % self.game.block_size
        self.position = deque()
        self.position.append([self.x, self.y])
        self.delta_x = self.game.block_size
        self.delta_y = 0
        self.food = 1
        self.eaten = False
        
    def reset(self):
        # start in the center of the screen
        self.x = self.game.width // 2
        self.y = self.game.height // 2
        # mod by grid size so its grid aligned.
        self.x = self.x - self.x % self.game.block_size
        self.y = self.y - self.y % self.game.block_size
        self.position = deque()
        self.position.append([self.x, self.y])
        self.delta_x = self.game.block_size
        self.delta_y = 0
        self.food = 1
        self.eaten = False

    def update_position(self, x, y):
        if x != self.position[-1][0] or y != self.position[-1][1]:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = x
            self.position[-1][1] = y

    def move(self, move, x, y, game, food):
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
            self.position.append([x, y])
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
            self.delta_y = -self.delta_x
            self.delta_x = 0
        elif np.array_equal(move, [0, 0, 1]) and self.delta_x == 0:
            # left - going vertical
            self.delta_x = self.delta_y
            self.delta_y = 0
        self.x = x + self.delta_x
        self.y = y + self.delta_y

        if self.x < game.margin or self.x > game.width - 2*game.margin \
                or self.y < game.margin \
                or self.y > game.height - 2*game.margin \
                or [self.x, self.y] in self.position:
            game.crash = True

        if self.x == food.x_food and self.y == food.y_food:
            food.next_food(game, self)
            self.eaten = True
            game.score = game.score + 1

        self.update_position(self.x, self.y)

    def display_player(self, game):
        if not game.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                if i == 0:
                    game.gameDisplay.blit(self.head_image,(x_temp,y_temp))
                else:
                    game.gameDisplay.blit(self.image, (x_temp, y_temp))

            update_screen()
        else:
            pygame.time.wait(300)


class Food:
    def __init__(self, game):
        self.game = game
        self.image = pygame.image.load("assets/apple.jpg")
        self.image = pygame.transform.scale(self.image, (game.margin, game.margin))
        # mod by grid size so its grid aligned.
        self.x_food = random.randint(game.margin, game.width - game.margin)
        self.x_food = self.x_food - self.x_food % game.block_size
        self.y_food = random.randint(game.margin, game.height - game.margin)
        self.y_food = self.y_food - self.y_food % game.block_size

    def reset(self):
        # mod by grid size so its grid aligned.
        self.x_food = random.randint(self.game.margin, self.game.width - self.game.margin)
        self.x_food = self.x_food - self.x_food % self.game.block_size
        self.y_food = random.randint(self.game.margin, self.game.height - self.game.margin)
        self.y_food = self.y_food - self.y_food % self.game.block_size

    def next_food(self, game, player):
        self.x_food = random.randint(game.margin, game.width - game.margin)
        self.x_food = self.x_food - self.x_food % game.block_size

        self.y_food = random.randint(game.margin, game.height - game.margin)
        self.y_food = self.y_food - self.y_food % game.block_size
        if (self.x_food, self.y_food) not in player.position:
            return self.x_food, self.y_food
        else:
            self.next_food(game, player)

    def display_food(self, game):
        game.gameDisplay.blit(self.image, (self.x_food, self.y_food))
        update_screen()


def run_game(speed):
    pygame.init()
    main = True
    game = Snake(1000, 1000, 20)
    player = game.player
    food = game.food
    record = 0
    # init move
    player.move([1, 0, 0], player.x, player.y, game, food)
    display(player, food, game, record)
    while main:
        move = [1, 0, 0]
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()
                main = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    move = [0, 0, 1]
                if event.key == pygame.K_RIGHT or event.key == ord('d'):
                    move = [0, 1, 0]

            if event.type == pygame.KEYUP:
                if event.key == ord('q'):
                    pygame.quit()
                    sys.exit()
                    main = False
                if event.key == ord('r'):
                    game.reset()
                    

        record = get_record(game.score, record)
        player.move(move, player.x, player.y, game, food)
        display(player, food, game, record)
        pygame.time.wait(speed)


if __name__ == '__main__':
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=10)
    args = parser.parse_args()
    run_game(100 - args.speed)
