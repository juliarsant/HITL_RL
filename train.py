"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""

import pygame
import time
from test import test
from model import ActorCritic
import torch
import torch.optim as optim
from lunar_lander import LunarLander 
import random

def train(gamma, lr, eps, steps, title):
    render = False #rendering
    betas = (0.9, 0.999)
    epsilon = 0.1

    env = LunarLander() #modified LunarLander
    
    policy = ActorCritic(11) #11 state elements in modified LunarLander
    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=betas)
    rewards = []
    steps_sum = []
    
    running_reward = 0

    for i_episode in range(0, eps):
        episode_rewards = 0

        if render and (i_episode%20 == 0):
            env = LunarLander(render_mode="human")
        else:
            env = LunarLander()

        state, _ = env.reset()

        for t in range(steps):
            random_num = random.random()

            if random_num < epsilon:
                action = random.randint(0,3)
            else:
                action = policy(state, False)

            state, reward, done, _, win = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward
            episode_rewards += reward

            if render and (i_episode > 500 and i_episode%20 == 0):
                env.render()
            if done:
                break

        # rewards.append(episode_rewards)
        # steps_sum.append(t)
                    
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
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > eps-1:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        
        if average_reward > 210:
            torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
            print("########## Solved! ##########")
            #test(name='LunarLander_{}}.pth')#.format(title))
            break
        

    return rewards, steps_sum

if __name__ == "__main__":
    train(0.99, 0.02, 10000, 800, "MODIFIED_JS1") #Gamma, lr, episodes, steps, path name


#random_seed = 543
#torch.manual_seed(random_seed)
#env = gym.make('LunarLander-v2')
#env.seed(random_seed)