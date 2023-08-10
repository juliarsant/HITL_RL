"""
Julia Santaniello
06/25/23

For training both the original and modified lunar lander games side by side
Input hyperparamters and the name of the folder to save the data in the run()
function.
"""

from test import test, mid_training_test
from model import ActorCritic
import torch
import torch.optim as optim
from lunar_lander import LunarLander
import random
from data import data_collection_versions
import gym
import pygame
import time

def train(gamma, lr, eps, steps, mod, title):
    epsilon = 0.1
    render = False
    human = False
    betas = (0.9, 0.999)
    argmax = False
    
    if not mod:
        #random_seed = 543
        #seed = torch.manual_seed(random_seed)
        env = gym.make('LunarLander-v2')
        #env.seed(seed)
        policy = ActorCritic(8)
    elif mod:
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

        #reset state
        state, _ = env.reset()

        for t in range(steps): #timesteps
            random_num = random.random() #random number to compare to epsilon

            if random_num < epsilon:
                action = random.randint(0,3)
            else:
                action = policy(state, argmax)

            

            state, reward, done, _, win = env.step(action)
            policy.rewards.append(reward)
            running_reward += reward #to average every 20 episodes
            episode_rewards += reward #Reward for every episode

            if render and (i_episode%20 == 0):
                env.render()
            if done:
                break
        if i_episode%20 == 0:
            mean, std, accuracy = mid_training_test(policy, mod, argmax)
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

        if average_reward > 200:
            argmax = True
            

        # if average_reward > 210:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        #     print("########## Solved! ##########")
        #     test(name='LunarLander_{}}.pth')#.format(title))
        #     break
        
    training_data = [training_mean, training_std, training_accuracy]
    return rewards, steps_sum, training_data

def run(trial:int):
    parameters = [
        [0.99, 0.01, 5000, 1000, True], #gamma, learning rate, episodes, steps, game version
        [0.99, 0.01, 5000, 1000, False]
    ]
    folder = "PreExperiment6"
    
    rewards1, steps1, training_data1 = train(parameters[0][0],parameters[0][1],parameters[0][2],parameters[0][3], parameters[0][4],"MODIFIED_lr001")
    print(training_data1)
    
    rewards2, steps2, training_data2 = train(parameters[1][0],parameters[1][1],parameters[1][2],parameters[1][3], parameters[1][4],"MODIFIED_lr002")
    
    data_collection_versions(rewards1, rewards2, steps1, steps2, training_data1, training_data2, folder, trial)


if __name__ == '__main__':
    for i in range(15):
        run(i)

"""
The success fluctuates, it will reach a peak, try something new, 
    and then go down a hill. I believe this is what makes the averages 
    over 20 trials lower
3000 is the peak for the average, but it must go farther most times 
    to find a winning policy. It might not hit 200 before 3000
Accuracy is wrong in the original lunar lander because it doesn't finish,
    it can hit a 200 reward and sit between the flags, but the game doesnt 
    stop. I didn't change anything in the original OpenAi gym implementation
Possibilities: could we change epislon which is currently at 0.1, 
    could we change epsilon when the average reward is above 200,
    could we do the best action when it gets above 200
Policy: The policy selects the action based on a sample of a 
    categorial distribution of the actions. I think that means that it takes
    the probabilities and uses them to make a decision. Not the best, but 
    based on the prob numbers
"""