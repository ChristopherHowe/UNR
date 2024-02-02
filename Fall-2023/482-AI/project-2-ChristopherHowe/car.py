#!/usr/bin/python3
import argparse
import logging
import sys
import os
import time
import numpy as np
import yaml
import random

import gym
from gym import wrappers, logger


defaultConfig = {
'numEpisodes': 5001,
'numBins': 9,
'alpha': 0.3,
'alphaDecay': 0.996,
'gamma': 0.5,
'epsilonStart': 0.9,
'epsilonEnd': 0.05,
'epsilonEndEpisode': 10000

}

class MountainCar:
    def __init__(self, env_id, train, test, model, render=False):
        config=None
        with open('car_config.yaml', 'r') as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)['config']
        # state =  [pos(x), vel(xdot)]
        self.min_vals = [-1.2,     -0.07]  
        self.max_vals = [0.6,     0.07]  
        self.num_bins = [config['numBins'], config['numBins']]  # This needs to be changed
        self.bins = np.array([np.linspace(self.min_vals[i], self.max_vals[i], self.num_bins[i])
                        for i in range(len(self.max_vals))])
        self.env_id = env_id
        self.train = train
        self.test = test
        self.model = model
        self.render = render
        self.config = config
        print(self.config)

    ################################################################################
    # CS482: this is the function that changes the continuous values of the state 
    ################################################################################
    def discretize_state(self,states):
        discretized_states = 0
        for i, state in enumerate(states):
            discretized_states += ((self.num_bins[i]+1)**i)*np.digitize(state, self.bins[i])
        return discretized_states

    
    def run(self):

        if not self.train and not self.test:
            print("[Warning] Specify train or test flag")
            print("for eg: python3 cart.py --train")
            print("or python3 cart.py --test --model cart.npy")

        if self.test:
            assert self.model is not None,\
                "Error: path to learned model path is not provided."
            if not os.path.exists(self.model):
                print("[Error] invalid model path\nNo such file as '" +
                    self.model+"' found")
                sys.exit(1)


        if self.render:
            env = gym.make(self.env_id, render_mode='human')
        else:
            env = gym.make(self.env_id)


        ############################################################################
        # CS482: This initial Q-table size should change to fit the number of
        # actions (columns) and the number of observations (rows)
        ############################################################################

        if self.train:
            # initialize Q table with zeros
            Q = np.zeros([(self.num_bins[0]+1)*(self.num_bins[1]+1)-1, env.action_space.n])
        if self.test:
            # load the saved model(learned Q table)
            Q = np.load(self.model, allow_pickle=True)

        ############################################################################
        # CS482: Here are some of the RL parameters. You have to tune the
        # learning rate (alpha) and the discount factor (gamma)
        ############################################################################

        alpha = self.config['alpha']
        gamma = self.config['gamma']
        
        # epsion-greedy params
        eps_start = self.config['epsilonStart']
        eps_end = self.config['epsilonEnd']
        # eps decay is based on which episode you want eps to reach the end value
        eps_decay = (eps_start - eps_end) / self.config['epsilonEndEpisode']
        eps=eps_start

        if self.train:
            n_episodes = self.config['numEpisodes']
            fails=0
            successes=0

            time_ = 1
            for episode in range(n_episodes):
                tick = 0
                reward = 0
                terminated=False
                truncated=False
                # env.reset() returns a tuple of internal state
                state = env.reset()
                s = self.discretize_state(state[0])
                print(f"episode {episode + 1}")

                if eps > eps_end:
                    eps -= eps_decay

                while not terminated and not truncated:
                    time_ += 1
                    tick += 1
                    action = 0
                    ri = -999
                    ################################################################
                    # CS482: Implement epsilon-greedy strategy that chooses 
                    # actions based on exploration or exploitation phase.
                    ###############################################################
                    p= random.random()
                    if p < eps:
                        action = np.random.randint(env.action_space.n)
                    else:
                        action = np.argmax(Q[s])
                    
                    state, reward, terminated, truncated, info = env.step(action)
                    sprime = self.discretize_state(state)
                    predicted_value = np.max(Q[sprime])

                    ################################################################
                    # CS482: Implement the update rule for Q learning here
                    ################################################################
                    Q[s, action] += alpha*(reward + gamma * predicted_value - Q[s, action])
                    s = sprime

                if episode % 10 == 0 and self.render:
                    env.render()
                    time.sleep(0.5)
                if episode % 1000 == 0:
                    alpha *= self.config['alphaDecay']
                    print(f"Alpha: {alpha} eps: {eps}")

                if state[0] < 0.5:
                    # print("fail ")
                    fails += 1
                else:
                    # print("success")
                    successes += 1

            print(f"Successes: {successes} Fails {fails}")
            np.save('car.npy', Q)
            print("Q table saved")

        if self.test:
            ########################################################################
            # CS482: this part of code relates to testing the performance of
            # the loaded (possibly learned) model
            ########################################################################
            reward=0
            terminated=False
            truncated=False
            state = env.reset()
            s = self.discretize_state(state[0])
            all_states = []
            while not terminated and not truncated:
                action = 0
                ri = -999
                # select action that yields max q value
                for q in range(env.action_space.n):
                    if Q[s][q] > ri:
                        action = q
                        ri = Q[s][q]
                state, reward, terminated, truncated, info = env.step(action)
                sprime = self.discretize_state(state)
                # render the graphics
                if self.render:
                    env.render()
                    time.sleep(0.005)

                all_states.append(state)

                s = sprime
            
            
            all_states = np.array(all_states)
            max_value = np.max(all_states[:, 0])
            print(f"Max: {max_value}")
            if state[0] >= 0.5:
                print('Success!')
            assert isinstance(all_states, np.ndarray)
            return all_states

def parse_args():

    parser = argparse.ArgumentParser(description=None)

    parser.add_argument('--env_id', dest='env_id', nargs='?',
                        default='MountainCar-v0', help='Select the environment to run')
    parser.add_argument('--train', dest='train', action='store_true',
                        help=' boolean flag to start training (if this flag is set)')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='boolean flag to start testing (if this flag is set)')
    parser.add_argument('--model', dest='model', type=str, default=None,
                        help='path to learned model')
    parser.add_argument('--render', dest='render', action='store_true',
                        help=' determine if the test/training session should be rendered')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    mountain_car = MountainCar(args.env_id, args.train, args.test, args.model, args.render)
    mountain_car.run()

if __name__ == '__main__':
    main()
