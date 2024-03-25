from experience_replay import PrioritizedExperienceReplay
import numpy as np
from numpy import clip
from numpy.random import rand, randint
from atari_preprocessing import ProcessedAtariEnv
from deep_q_networks import DeepQNetwork
import tensorflow as tf
from tensorflow import math
argmax = math.argmax
import time
import os
import gym 
from log_training import QLearningLogger

class Wrapper (gym.Env):

    def __init__(self, game_id):
        self.game = Game(game_id)
        
    def reset(self):
        self.game.reset()
        return self.game.get_observation()
    
    def step(self, action):
        # Take a step in the game based on the given action
        reward, done = self.game.take_action(action)
        observation = self.game.get_observation()
        return observation, reward, done, {}
    
    def render(self, mode='human'):
        # Render the current state of the game
        self.game.render()
    
    def close(self):
        # Clean up resources if necessary
        pass

