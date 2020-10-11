"""
Created by Edward Li at 10/7/20
following training from (https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf)
"""
import argparse
import datetime
import os
import random
from itertools import count

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygame
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from DQN import DQNCNN, DQNFCN
from replay_memory import Transition, ReplayMemory
from snake import Snake, display, get_record, __num_actions__


def get_run_name():
    """
     Function to return wandb run name.
     Currently this is the only way to get the run name.
     In a future bugfix wandb.run.name will then be available.
    :return: run name or dryrun when dryrun.
    """
    try:
        wandb.run.save()
        run_name = wandb.run.name
        return run_name
    except BaseException:
        # when WANDB_MODE=dryrun the above will throw an exception. At that
        # stage we except and return "dryrun"
        return "dryrun"


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    pass
plt.ion()


def init_wandb(args, tag):
    # initialize weights and biases.
    wandb.init(project="Snake-RL", dir="./.wandb/", tags=[tag])
    wandb.tensorboard.patch(save=False, tensorboardX=False)
    wandb.config.update(args)


global_steps = 0


def select_action(state, policy_net, n_actions, eps_start, eps_end, eps_decay):
    global global_steps
    sample = random.random()
    # exponential decay
    # eps_threshold = eps_end + (eps_start - eps_end) * \
    #                 math.exp(-1. * global_steps / eps_decay)
    # linear epsilon decay
    eps_threshold = max((eps_start - global_steps * eps_decay), eps_end)

    global_steps += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(args, memory, policy_net, target_net, optimizer):
    if len(memory) < args.batch_size:
        return None
    transitions = memory.sample(args.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(args.batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.item()


def get_env_state(image, game, state_scale):
    image = cv2.resize(image[:-40, :, ::-1] / 255,
                       (game.width * state_scale // game.block_size, game.height * state_scale // game.block_size),
                       cv2.INTER_AREA)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return image, image_tensor


def get_state_cnn_from_env(game):
    state = game.get_state_cnn()
    # test_im = state.copy()
    # test_im[test_im==1] = 0.465
    # test_im[test_im==2] = 0.50
    # test_im[test_im ==3] = 0.80
    # test_im[test_im ==4] = 0.30
    # cv2.imshow("t",test_im)
    # cv2.waitKey(1)
    state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    return state, state_tensor


def get_state_fcn_from_env(game):
    state = game.get_state_fcn()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
    return state, state_tensor


def get_reward(player, food, game, rewards):
    if game.crash:
        return rewards[1]
    if player.eaten:
        return rewards[0]
    # the diagonal length is the longest in the grid
    dist_reward_scale = 10 if len(rewards) < 3 else rewards[2]
    dist = np.sqrt((food.x_food - player.x) ** 2 + (food.y_food - player.y) ** 2) / game.block_size
    # velocity = np.sqrt((player.x - player.prev_x) ** 2 + (player.y - player.prev_y) ** 2) / game.block_size
    # based on distance to food, reward the agent
    dist_reward = 1 - np.power(dist, 0.4)

    reward = dist_reward * dist_reward_scale
    return reward


def save_ckpt(output_dir, name, model):
    """
    save checkpoint to output directory
    :param output_dir: output directory
    :param name: checkpoint name
    :param model: model
    :return:
    """
    model_file = "{}/model_{}.pth".format(output_dir, name)
    torch.save(model.state_dict(), model_file)
    wandb.save(model_file)


def main(args):
    init_wandb(args, tag="snake RL")
    now = datetime.datetime.now()
    run_name = get_run_name()
    logger = SummaryWriter(log_dir="./.runs/{}_{}".format(now.strftime("%Y-%m-%d_%H_%M"), run_name))
    output_dir = os.path.join(args.output_dir, "{}_{}".format(now.strftime("%Y-%m-%d_%H_%M"), run_name))
    os.makedirs(output_dir, exist_ok=True)
    pygame.font.init()
    pygame.init()
    if args.training_type.lower() == "cnn":
        policy_net = DQNCNN(args.screen_size * args.state_scale // args.block_size, Snake.num_actions, in_channels=1,
                            features=args.features).to(
            device)
        target_net = DQNCNN(args.screen_size * args.state_scale // args.block_size, Snake.num_actions, in_channels=1,
                            features=args.features).to(
            device)
        get_state = get_state_cnn_from_env
    elif args.training_type.lower() == "fcn":
        policy_net = DQNFCN(Snake.fcn_state_size, Snake.num_actions,
                            features=args.features).to(
            device)
        target_net = DQNFCN(Snake.fcn_state_size, Snake.num_actions,
                            features=args.features).to(
            device)
        get_state = get_state_fcn_from_env
    else:
        raise ValueError("network type not supportted")

    wandb.watch(policy_net)

    if args.pretrained is not None:
        policy_net.load_state_dict(torch.load(args.pretrained))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # use RMSprop here because there is uncertainty what momentum does in RL environment.
    optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr)
    memory = ReplayMemory(args.memory_size)
    record = 0
    game = Snake(args.screen_size, args.screen_size, block_size=args.block_size, state_scale=args.state_scale)
    player = game.player
    food = game.food
    # init move
    player.move([1, 0, 0], player.x, player.y, game, food)
    for epi in tqdm(range(args.episodes)):
        if epi > int(args.episodes * 0.8):
            args.show_game = True
        game.reset()
        state_image, state = get_state(game)
        # get_env_state(display(player, food, game, record), game, args.state_scale)
        if args.show_game:
            display(player, food, game, record)

        score = 0
        episode_loss = 0

        total_reward = 0

        vid_writer = None
        if epi % args.save_interval == 0:
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            output_file = os.path.join(output_dir, "episode_{}_gamplay.mp4".format(epi))
            screen_x, screen_y = game.game_display.get_size()
            vid_writer = cv2.VideoWriter(output_file, fourcc, 30, (screen_x, screen_y))

        for t in count():
            # select action based on state
            action = select_action(state, policy_net, __num_actions__, *args.eps)
            move = game.actions[action.item()]
            player.move(move, player.x, player.y, game, food)
            reward = get_reward(player, food, game, args.reward)
            reward_tensor = torch.tensor([reward], dtype=torch.float, device=device)

            score = game.score
            # observer our new state
            next_image, next_state = get_state(game)

            if args.show_game:
                display(player, food, game, record)

            if vid_writer is not None:
                image = display(player, food, game, record)[:, :, ::-1]
                vid_writer.write(image)

            if game.crash:
                next_state = None

            # store the transition in memory
            memory.push(state, action, next_state, reward_tensor)

            # move to next state
            state = next_state

            # perform one optimization step (on the target network)
            loss = optimize_model(args, memory, policy_net, target_net, optimizer)
            if loss is not None:
                episode_loss += loss

            total_reward += reward

            if game.crash:
                logger.add_scalar("game record", score, global_step=epi)
                logger.add_scalar("game duration", t + 1, global_step=epi)
                logger.add_scalar("avg_reward", total_reward / (t + 1), global_step=epi)
                logger.add_scalar("total_reward", total_reward, global_step=epi)
                logger.add_scalar("episode loss", episode_loss / (t + 1), global_step=epi)
                break

        new_record = get_record(score, record)

        if new_record > record:
            record = new_record
            save_ckpt(output_dir, "policy_net", policy_net)
        # Update the target network, copying all weights and biases in DQN
        if global_steps % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if epi % args.save_interval == 0:
            assert vid_writer is not None
            vid_writer.release()
            wandb.log({"video": wandb.Video(output_file,format="mp4",fps=30)})
            image = display(player, food, game, record)
            cv2.imwrite(os.path.join(output_dir, "episode_{}_end_img.jpg"), image[:, :, ::-1])
            logger.add_image("episode_end_img", torch.from_numpy(image).permute(2, 0, 1), global_step=epi)
            save_ckpt(output_dir, "episode_{}_ckpt".format(epi), policy_net)

    print("Complete")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("snake-rl training parser")
    parser.add_argument("--speed", help="game_speed", type=int, default=10)
    parser.add_argument("--show_game", help="show game visuals?", action="store_true")
    parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
    parser.add_argument("--training_type", help="type of training,(fcn, cnn)",default="cnn")
    parser.add_argument("--lr", help="learning rate", type=float, default=0.0003)
    parser.add_argument("--gamma", help="gamma, for balancing near/long term reward", type=float, default=0.999)
    parser.add_argument("--memory_size", help="size of memory", type=int, default=10000)
    parser.add_argument("--eps", help="epsilon, in 'start, end, decay' format", default="0.9,0.05,200")
    parser.add_argument("--target_update", help="duration before update", default=10, type=int)
    parser.add_argument("--features", help="network layer feature size, in 'f1,f2,f3' format", default="32,64,128")
    parser.add_argument("--episodes", help="number of training episodes", default=1000, type=int)
    parser.add_argument("--screen_size", help="screen size", default=400, type=int)
    parser.add_argument("--block_size", help="game block_size", default=20, type=int)
    parser.add_argument("--state_scale", help="scale of downsized state image", default=1, type=int)
    parser.add_argument("--save_interval", help="interval between saving weights", default=100, type=int)
    parser.add_argument("--pretrained", help="pre trained checkpoint", default=None)
    parser.add_argument("--reward", help="reward for eating,dying", default="100,-10")
    parser.add_argument("--output_dir", help="output directory", default="./output")
    args = parser.parse_args()
    args.eps = [float(x) for x in args.eps.split(",")]
    args.conv_features = [int(x) for x in args.conv_features.split(",")]
    args.reward = [float(x) for x in args.reward.split(",")]
    main(args)
