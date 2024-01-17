"""
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

Variables used throughout the program.
"""
from games.lunar_lander import LunarLander
from games.car_racing import CarRacing
from games.snake_game import SnakeEnv

environment = SnakeEnv
episodes = 10000
steps = 600
trials = 1
gamma = 0.99
learning_rate = 0.1
greedy_e_epsilon = 0.15
alpha = 0.1
seed = 10
obs_size_values = 11
num_actions = 2
run_demos = False
algorithm_name = "simplePG"
env_name = "car_racing"
exploration_type = None
demo_name_hitl = "pilot_001H_12202023"
demo_name = "pilot_001H_12202023"
graph_path = "./data/results/graphs/"
num_demos = 75
data_folder_name = "./data/results/"

