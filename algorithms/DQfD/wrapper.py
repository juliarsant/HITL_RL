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
from log_training import QLearningLogger

# func design 
# take in game id and output a game 
# 
def wrapper():
    print ("in progress")