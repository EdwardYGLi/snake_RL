"""
Created by Edward Li at 10/6/20
followed game code from https://github.com/maurock/snake-ga
"""

import argparse
import random
import sys
from collections import deque

import numpy as np
import cv2
import pygame

__num_actions__ = 3


def update_screen():
    pygame.display.update()


def display(player, food, game, record):
    game.game_display.fill((255, 255, 255))
    display_ui(game, game.score, record)
    player.display_player(game)
    food.display_food(game)
    image = pygame.surfarray.array3d(pygame.display.get_surface()).swapaxes(0, 1)
    return image


def display_ui(game, score, record):
    myfont = pygame.font.SysFont('Segoe UI', 20)
    myfont_bold = pygame.font.SysFont('Segoe UI', 20, True)
    text_score = myfont.render('SCORE: ', True, (0, 0, 0))
    text_score_number = myfont.render(str(score), True, (0, 0, 0))
    text_highest = myfont.render('HIGHEST SCORE: ', True, (0, 0, 0))
    text_highest_number = myfont_bold.render(str(record), True, (0, 0, 0))
    game.game_display.blit(text_score, (45, game.height + 20))
    game.game_display.blit(text_score_number, (120, game.height + 20))
    game.game_display.blit(text_highest, (190, game.height + 20))
    game.game_display.blit(text_highest_number, (350, game.height + 20))
    new_surf = pygame.pixelcopy.make_surface(game.bg)
    game.game_display.blit(new_surf,(0,0))


def get_record(score, record):
    if score >= record:
        return score
    else:
        return record


class Snake:
    def __init__(self, width, height, block_size=20,state_scale = 1):
        pygame.display.set_caption("Snake_RL")
        self.width = width
        self.height = height
        self.state_w = width//block_size * state_scale
        self.state_h = height//block_size * state_scale
        self.state_scale = state_scale
        self.bg = np.ones((width, height, 3), dtype=np.uint8) * 255
        self.bg[:block_size, :, :] = 0
        self.bg[:, :block_size, :] = 0
        self.bg[-block_size:, :, :] = 0
        self.bg[:, -block_size:, :] = 0
        self.bg = self.bg.swapaxes(0,1)
        self.diagonal = np.sqrt((self.state_w - 1) ** 2 + (self.state_h - 1) ** 2)

        self.game_display = pygame.display.set_mode((width, height + 40))
        self.game_buffer = None
        self.score = 0
        self.crash = False
        self.block_size = block_size
        self.player = Player(self)
        self.food = Food(self)
        self.actions = {
            0: np.array([1, 0, 0]),
            1: np.array([0, 1, 0]),
            2: np.array([0, 0, 1])
        }

    def get_state_cnn(self):
        state = np.zeros((self.state_h,self.state_w,1),dtype=np.float)
        # draw borders
        state[:self.state_scale,:,:] = 4
        state[:,:self.state_scale,:] = 4
        state[-self.state_scale:,:,:] = 4
        state[:,-self.state_scale:,:] = 4

        state = self.player.update_state(state,self)
        state = self.food.update_state(state,self)
        return state

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
        self.prev_x = self.x
        self.prev_y = self.y

        self.position = deque()
        self.position.append([self.x, self.y])
        # self.position.extend(
        #     [[self.x - 2 * game.block_size, self.y], [self.x - game.block_size, self.y], [self.x, self.y]])
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
                self.position.append(self.position.popleft())

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
        
        self.prev_x = self.x
        self.prev_y = self.y

        self.x = x + self.delta_x
        self.y = y + self.delta_y

        if self.x < game.block_size or self.x > game.width - 2 * game.block_size \
                or self.y < game.block_size \
                or self.y > game.height - 2 * game.block_size \
                or [self.x, self.y] in self.position:
            game.crash = True

        if self.x == food.x_food and self.y == food.y_food:
            food.next_food(game, self)
            self.eaten = True
            game.score = game.score + 1

        self.update_position(self.x, self.y)

    def update_state(self,state,game):
        if not game.crash:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                x_temp = x_temp//game.block_size * game.state_scale
                y_temp = y_temp//game.block_size * game.state_scale
                if i == 0:
                    state[y_temp,x_temp] = 1
                else:
                    state[y_temp,x_temp] = 2
        return state

    def display_player(self, game):
        # if not game.crash:
        for i in range(self.food):
            x_temp, y_temp = self.position[len(self.position) - 1 - i]
            if i == 0:
                game.game_display.blit(self.head_image, (x_temp, y_temp))
            else:
                game.game_display.blit(self.image, (x_temp, y_temp))
        update_screen()
        # else:
        #     pygame.time.wait(300)


class Food:
    def __init__(self, game):
        self.game = game
        self.image = pygame.image.load("assets/apple.png")
        self.image = pygame.transform.scale(self.image, (game.block_size, game.block_size))
        # mod by grid size so its grid aligned.
        self.x_food = random.randint(game.block_size, game.width - 2 * game.block_size)
        self.x_food = self.x_food - self.x_food % game.block_size
        self.y_food = random.randint(game.block_size, game.height - 2 * game.block_size)
        self.y_food = self.y_food - self.y_food % game.block_size

    def reset(self):
        # mod by grid size so its grid aligned.
        self.x_food = random.randint(self.game.block_size, self.game.width - 2 * self.game.block_size)
        self.x_food = self.x_food - self.x_food % self.game.block_size
        self.y_food = random.randint(self.game.block_size, self.game.height - 2 * self.game.block_size)
        self.y_food = self.y_food - self.y_food % self.game.block_size

    def next_food(self, game, player):
        self.x_food = random.randint(game.block_size, game.width - 2 * game.block_size)
        self.x_food = self.x_food - self.x_food % game.block_size

        self.y_food = random.randint(game.block_size, game.height - 2 * game.block_size)
        self.y_food = self.y_food - self.y_food % game.block_size
        if (self.x_food, self.y_food) not in player.position:
            return self.x_food, self.y_food
        else:
            self.next_food(game, player)

    def display_food(self, game):
        game.game_display.blit(self.image, (self.x_food, self.y_food))
        update_screen()

    def update_state(self,state,game):
        x = self.x_food//game.block_size * game.state_scale
        y = self.y_food//game.block_size * game.state_scale
        state[y,x,:] = 3
        return state


def run_game(speed):
    # Set options to activate or deactivate the game view, and its speed
    pygame.font.init()
    pygame.init()
    main = True
    game = Snake(500, 500, 20)
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
                pygame.quit()
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
        img = display(player, food, game, record)
        pygame.time.wait(speed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=int, default=1)
    args = parser.parse_args()
    run_game(100 - args.speed)
