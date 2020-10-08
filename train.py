"""
Created by Edward Li at 10/7/20
"""
import argparse
import datetime
import math
import random
import pygame
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import torch
import torch.functional as F
import torch.optim as optim
import wandb
from itertools import count
from tqdm import tqdm
from snake import Snake,Player,Food,display,get_record, __num_actions__
from DQN import DQN
from replay_memory import Transition, ReplayMemory


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
    wandb.init(project="Snake-RL", dir="../wandb/", tags=[tag])
    wandb.tensorboard.patch(save=True, tensorboardX=True)
    wandb.config.update(args)


def select_action(state, policy_net, n_actions, eps_start,eps_end, eps_decay):
    global global_steps 
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * \
                    math.exp(-1. * global_steps / eps_decay)
    global_steps += 1
    action = torch.tensor(np.zeros(n_actions), device=device, dtype=torch.float)
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            max_ind = policy_net(state).max(1)[1]
            action[max_ind] = 1
            return action
    else:
        ind = random.randrange(n_actions)
        action[ind] = 1
        return action

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def optimize_model(args, memory, policy_net, target_net,optimizer):
    if len(memory) < args.batch_size:
        return
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


def get_env_state(image, game):
    image = cv2.resize(image[:-40, :, ::-1],(game.width // game.block_size, game.height // game.block_size), cv2.INTER_AREA)
    image_tensor = torch.tensor(image).unsqueeze(0).to(device)
    return image, image_tensor


def get_reward(player, crash, rewards):
    reward = 0
    if crash:
        reward = rewards[1]
        return reward
    if player.eaten:
        reward = reward[0]
    return reward


def main(args):
    now = datetime.datetime.now()
    run_name = get_run_name()
    logger = SummaryWriter(log_dir="../.runs/{}_{}".format(now.strftime("%Y-%m-%d_%H_%M"), run_name))
    output_dir = os.path.join(args.output_dir, "{}_{}".format(now.strftime("%Y-%m-%d_%H_%M"), run_name))
    os.makedirs(output_dir, exist_ok=True)
    pygame.font.init()
    pygame.init()
    policy_net = DQN(args.screen_size//args.block_size,__num_actions__,args.conv_features)
    target_net = DQN(args.screen_size//args.block_size,__num_actions__,args.conv_features)
    if args.pretrained is not None:
        policy_net.load_state_dict(torch.load(args.pretrained))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # use RMSprop here because there is uncertainty what momentum does in RL environment.
    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(10000)
    global_steps = 0
    score_plot = []
    durations = []
    record = 0
    for epi in tqdm(range(args.episodes)):
        game = Snake(args.screen_size,args.screen_size, block_size=20,margin=20)
        player = game.player
        food = game.food
        # init move
        player.move([1, 0, 0], player.x, player.y, game, food)
        last_image, last_state = get_env_state(display(player, food, game, record))
        curr_state = last_state
        state = last_state - curr_state
        for t in count():
            # select action based on state
            action = select_action(last_state,policy_net,__num_actions__,*args.eps)

            player.move(action.detach().numpy(),player.x,player.y,game,food)
            reward = torch.tensor([get_reward(player,game.crash,args.reward)],device=device)
            record = get_record(game.score, record)

            # observer our new state
            last_state = curr_state
            curr_image, curr_state = get_env_state(display(player,food,game,record))
            if not game.crash:
                next_state = curr_state - last_state
            else:
                next_state = None

            # store the transition in memory
            memory.push(state,action,next_state,reward)

            # move to next state
            state = next_state

            # perform one optimization step (on the target network)
            optimize_model(args,memory,policy_net,target_net,optimizer)

            if game.crash:
                score_plot.append(record)
                durations.append(t+1)
                
        # Update the target network, copying all weights and biases in DQN
        if epi % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        if epi % args.save_interval == 0:
            torch.save(policy_net.state_dict())

    print("Complete")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("snake-rl training parser")
    parser.add_argument("--speed", help="game_speed", type=int, default=10)
    parser.add_argument("--show_game", help="show_game", action="store_true")
    parser.add_argument("--batch_size", help="batch_size", type=int, default=128)
    parser.add_argument("--gamma", help="gamma, for balancing near/long term reward", type=float, default=0.999)
    parser.add_argument("--eps", help="epsilon, in 'start, end, decay' format", default="0.9,0.05,200")
    parser.add_argument("--target_update", help="duration before update", default=10)
    parser.add_argument("--conv_features", help="convolution feature size, in 'f1,f2,f3' format", default="32,64,128")
    parser.add_argument("--episodes", help="number of training episodes", default=1000)
    parser.add_argument("--screen_size", help="screen size", default=600)
    parser.add_argument("--block_size",help="game block_size", default=20)
    # parser.add_argument("--game_speed",help="game_speed (sleep duration)",type=int,default=30)
    parser.add_argument("--save_interval", help="interval between saving weights", default=10, type=int)
    parser.add_argument("--pretrained",help="pre trained checkpoint",default=None)
    parser.add_argument("--reward",help="reward for eating,dying",default="10,-1")
    parser.add_argument("--output_dir",help="output directory",default = "./output")
    args = parser.parse_args()
    args.eps = [float(x) for x in args.eps.split(",")]
    args.conv_features = [int(x) for x in args.conv_features.split(",")]
    args.reward = [int(x) for x in args.reward.split(",")]
    main(args)
