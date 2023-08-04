"""
Julia Santaniello
06/25/23

Can train multiple types of hyperparameters in one run
Just adjust the hyperparameters in the run() function
"""

from test import test, mid_training_test
from model import ActorCritic
import torch
import torch.optim as optim
from lunar_lander import LunarLander
import matplotlib.pyplot as plt
import numpy as np
import random
from data import data_collection 



def train(gamma, lr, eps, steps, title):
    # Defaults parameters:
    #    gamma = 0.99
    #    lr = 0.02
    #    betas = (0.9, 0.999)
    #    random_seed = 543
    epsilon = 0.1
    render = False
    betas = (0.9, 0.999)
    #random_seed = 543
    
    #torch.manual_seed(random_seed)
    #env = gym.make('LunarLander-v2')
    #env.seed(random_seed)
    env = LunarLander()
    
    policy = ActorCritic(11) #The neural network
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    print(lr,betas)
    rewards = []
    steps_sum = []
    training_mean = []
    training_std = []
    training_accuracy = []
    
    running_reward = 0
    for i_episode in range(0, eps): #episodes
        episode_rewards = 0
        #When to render
        if render and (i_episode%20 == 0):
            env = LunarLander(render_mode="human")
        else:
            env = LunarLander()

        #reset state
        state, _ = env.reset()

        for t in range(steps): #timesteps
            random_num = random.random() #random number to compare to epsilon

            if random_num < epsilon:
                action = random.randint(0,3)
            else:
                action = policy(state, False)

            state, reward, done, _, win = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward #to average every 20 episodes
            episode_rewards += reward #Reward for every episode

            if render and (i_episode%20 == 0):
                env.render()
            if done:
                break
        if i_episode%20 == 0:
            mean, std, accuracy = mid_training_test(policy, True, False)
            training_mean.append(mean)
            training_std.append(std)
            training_accuracy.append(accuracy)

        rewards.append(episode_rewards)
        steps_sum.append(t)
                    
        # Updating the policy :
        optimizer.zero_grad()
        loss = policy.calculateLoss(gamma)
        loss.backward()
        optimizer.step()        
        policy.clearMemory()

        if i_episode % 20 == 0:
            average_reward = running_reward/20
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, average_reward))
            running_reward = 0
            

        # if average_reward > 210:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        #     print("########## Solved! ##########")
        #     test(name='LunarLander_{}}.pth')#.format(title))
        #     break
        
    training_data = [training_mean, training_std, training_accuracy]
    return rewards, steps_sum, training_data

def run(trial:int):
    parameters = [
        [0.99, 0.01, 10000, 1000], #gamma, learning rate, episodes, steps, game version
        [0.99, 0.01, 10000, 1000],
        [0.99, 0.01, 10000, 1000]
    ]
    folder = "PreExperiment3"
    
    rewards1, steps1, training_data1 = train(parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3], "MODIFIED_lr001")
    print("at 0.02")
    rewards2, steps2, training_data2 = train(parameters[1][0],parameters[1][1],parameters[1][2],parameters[1][3], "MODIFIED_lr002")
    print("at 0.03")
    rewards3, steps3, training_data3 = train(parameters[2][0],parameters[2][1],parameters[2][2],parameters[1][3], "MODIFIED_lr003")
    
    data_collection(rewards1, rewards2, rewards3, steps1, steps2, steps3, training_data1, training_data2, training_data3, folder, trial)


if __name__ == '__main__':
    for i in range(20):
        run(i)
