'''
This is the script that will start the training.
The algorithm itself is in ATOC_COMA.trainer
'''

import argparse
import gym
import numpy as np
import os
import tensorflow as tf
import time
import pickle

import ATOC_COMA.common.tf_util as U
from ATOC_COMA.trainer.atoc_coma import ATOC_COMA_AgentTrainer
import tensorflow.contrib.layers as layers
from tensorflow.contrib import rnn


def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=6000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")  # maddpg: global q function, ddpg: local q function
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='test', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/home/ubuntu/maddpg/saved_policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="/home/ubuntu/maddpg/generated_plots", help="directory where plot data is saved")
    return parser.parse_args()


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world. This will make the world according to the scenario see "simple_spread.py" > make_world
    world = scenario.make_world()
    # create multiagent environment. Now all the functions we need are in the env
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)  # reset, reward, obs are callbacks
    return env

'''
Since we want all the actors and critics to have the network, we want to share, i.e., "reuse" the parameters
Also, to do this we need to make the "scope" of the variables that we want to reuse the same.

look at variable_scope.py
'''

def mlp_actor_model_1(input, num_outputs, scope, reuse=True, num_units=64, rnn_cell=None):
    '''
    :param input: from local observations
    :param num_outputs: output of Actor_Part1
    :param scope: variable scope
    :param reuse: true/false for sharing variables
    :param num_units: hidden units
    :param rnn_cell: ?

    NOTE:
    Setting reuse "true" so that all agents can update the same model
    '''
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def mlp_actor_model_2(input, num_outputs, scope, reuse=True, num_units=64, rnn_Cell=None):
    '''
    :param input: output from the attentional network and comm. channel (if necessary)
    This model is designed to be the second part of the actor network
    '''
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs, activation_fn=None)
        return out
        # TODO: Check what the activation function be?


def mlp_critic_model(input, num_outputs, scope, reuse=True, num_units=128, rnn_cell=None):
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


def rnn_model(input, num_outputs, scope, reuse=False, num_units=128, length=10):
    # This is the Recurrent Neural Network for the attention unit. Binary classifier.
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        cell = rnn.BasicRNNCell(num_units=num_units, activation="tanh")


def lstm_model():
    pass


def train(arglist):
    with U.single_threaded_session():
        # Create environment. This takes the scenario and creates a world, and creates and environment with all the
        # functions required. This is similar to the AI Gym environment.
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers.
        # obs_shape_n is a list of the observation shape of each agent
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]    # env.n is the total number of agents
        num_adversaries = min(env.n, arglist.num_adversaries)

        # Initialize the Actor and Critic network (these are shared among all agents)
        actor1 = mlp_actor_model_1    # reference to the object class, i.e., not using ()
        actor2 = mlp_actor_model_2
        critic = mlp_critic_model
        attn = rnn_model
        trainer = ATOC_COMA_AgentTrainer("shared_trainer",    # Dont we need the same number of trainers as agents in the env?
                                         actor1,
                                         actor2,
                                         critic,
                                         attn,
                                         obs_shape_n,
                                         env.action_space,
                                         1,
                                         arglist,
                                         local_q_func=True)
        '''
        So what have we done till here:
        We have created the env based on the scenario.
        We have created the neural network models for the actor1, actor2, attention unit, and the critic. Remember, these
        are just the models. Also we are yet to define the LSTM!!!
        '''

