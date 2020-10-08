"""
Created by Edward Li at 10/7/20
"""
import argparse
import math
import random
import pygame
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.functional as F
import wandb
from tqdm import tqdm
from snake import Snake,Player,Food,display,get_record, __num_actions__
from DQN import DQN
from replay_memory import Transition, ReplayMemory



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


def select_action(state, policy_net, n_actions, eps_end, eps_start, eps_decay):
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
global_steps = 0


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


def main(args):
    pygame.font.init()
    pygame.init()
    policy_net = DQN(args.screen_size//args.block_size,__num_actions__,args.conv_features)
    target_net = DQN(args.screen_size//args.block_size,__num_actions__,args.conv_features)
    if args.pretrained is not None:
        policy_net.load_state_dict(torch.load(args.pretrained))
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    for epi in tqdm(args.episodes):

    game = Snake(args.screen_size,args.screen_size, block_size=20,margin=20)
    player = game.player
    food = game.food






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
    parser.add_argument("--game_speed",help="game_speed (sleep duration)",type=int,default=30)
    parser.add_argument("--eval_interval", help="interval between eval episodes", default=10, type=int)
    parser.add_argument("--pretrained",help="pre trained checkpoint",default=None)
    parser.add_argument("--reward",help="reward for eating,dying",default="10,-1")
    args = parser.parse_args()
    args.eps = [float(x) for x in args.eps.split(",")]
    args.conv_features = [int(x) for x in args.conv_features.split(",")]
    args.reward = [int(x) for x in args.reward.split(",")]
    main(args)
