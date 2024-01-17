
import numpy as np
import time
from PIL import Image
import pandas as pd
import pygame
from algorithms.simplePG import SimplePG
import demonstrations as demo
import pickle
import matplotlib.pyplot as plt
import csv
from imports import environment, demo_name_hitl, seed, run_demos, steps, exploration_type, greedy_e_epsilon, gamma, learning_rate, obs_size_values, num_actions,algorithm_name, env_name, episodes, num_demos, trials


"""
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

For training a single humna-in-the-loop policy. Saves results in Data folder.
"""
env = environment()
env.render()
demo_name = demo_name_hitl

"""
Starts training by initially preparing user to play
"""
def start():
    input("Press Enter to Start Demonstrations: ")
    print("Starting in 3...")
    time.sleep(1)
    print("2...")
    time.sleep(1)
    print("1...")
    time.sleep(1)
    print("Start!")
    print("")

"""
main function of hitl_train
"""
def run(trials, run_demos:bool):
    if run_demos:
        start()
        demo.main(demo_name)
        print("Thank you!")
        time.sleep(1)

    print("Starting agent training...")

    r,s,a=[],[],[]
    for i in range(trials):
        rewards, steps, accuracy = train_with_demonstrations()
        r.append(rewards)
        s.append(steps)
        a.append(accuracy)

    print("Done! Saving Data!")
    r = np.asarray(r).mean(axis=0)
    s = np.asarray(s).mean(axis=0)
    a = np.asarray(a).mean(axis=0)
    save_data(r, s, a)


"""
Saves training data to csv file
"""
def save_data(r,s,a):
    arr = np.asarray([r,s,a])
    df = pd.DataFrame(arr)
    df.to_csv("./data/results/{}.csv".format(demo_name))

"""
trains the agent with demonstrations and preferred number of demonstrations
"""
def train_with_demonstrations():
    with open('./data/demonstrations/{}.pickle'.format(demo_name), 'rb') as file:
        demo_dict = pickle.load(file)

    file.close()
    
    agent = SimplePG(num_actions=num_actions, input_size=obs_size_values, hidden_layer_size=12, learning_rate=learning_rate, decay_rate=0.99, gamma=gamma, greedy_e_epsilon=greedy_e_epsilon, random_seed=10)
    env = environment()
    

    demo_eps = len(demo_dict)-2

    avg_rewards_past, avg_steps_past, avg_accuracy_past = [], [], []
    sum_wins, running_reward, running_steps = 0,0,0
    eps = episodes + demo_eps
    ii = 0
    for i in range(eps):
        if i<num_demos:
            ii = np.random.randint(74)
            steps = demo_dict[ii]["steps"]
            seed = demo_dict[ii]["seed"]
            state = env.reset(seed=seed)
            human = True
        else:
            human= False
            steps = 600
            state = env.reset()

        episode_reward = 0

        for j in range(steps+1):
            done = False
            action = agent.pickAction(state, exploring=True)
            action_robot = action
            #pick an action
            if human:
                action = demo_dict[ii]["actions"][j]
                agent.save_human_action(action)
                reward = demo_dict[ii]["rewards"][j]
                state = demo_dict[ii]["states"][j]
                agent.saveHumanStep(state=state, reward=reward, action=action)
            
            next_state, reward, done, win= env.step(action)


                

            if j == steps + 1:
                reward = -100

            agent.saveStep(state=state, reward=reward, action=action, next_state=next_state, done=done)

            episode_reward += reward
            state = next_state

            if done or j == steps:
                running_steps += j
                running_reward += episode_reward
                if human:
                    ii += 1
                if win:
                    sum_wins += 1
                break
        
        if human:
            agent.finishEpisode(episode_reward, True)
        else:
            agent.finishEpisode(episode_reward, False)

        if exploration_type == "decay":
            agent._explore_eps -= 1/5000



        if i%20==0:
            avg_accuracy_past.append(sum_wins/20)
            avg_rewards_past.append(running_reward/20)
            avg_steps_past.append(running_steps/20)
            print('Episode {}\tlength: {}\treward: {}\t accuracy: {}'.format(i, j, running_reward/20, sum_wins/20))
            if sum_wins/20 > 0.50 and not human and i > 1000:
                break
            running_reward, running_steps, sum_wins = 0,0,0
    


        agent.updateParamters()
    
    assert(len(avg_rewards_past)==len(avg_steps_past)==len(avg_accuracy_past))

    save_policy(agent)

    return avg_rewards_past, avg_steps_past, avg_accuracy_past

def save_policy(agent):
    env = environment()
    env.render()

    for i in range(5):
        state = env.reset()
        reward_sum = 0

        for j in range(steps):
            action = agent.pickAction(state, exploring=False)

            #Return state, reward
            new_state, reward, done, win = env.step(action)
            reward_sum += reward
            state = new_state
            if done:
                print(reward_sum)
                print("win: ", win)
                break
    # with open('./data/policies/{}.pickle'.format("policy1"), 'wb') as handle:
    #         pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)   

def evaluate_policy():
    with open('./data/policies/{}.pickle'.format("policy1"), 'rb') as file:
        agent = pickle.load(file)
    env = environment()
    env.render()

    for i in range(5):
        state = env.reset()

        for j in range(steps):
            action = agent.pickAction(state, exploring=False)

            #Return state, reward
            new_state, reward, done, _ = env.step(action)

            state = new_state
            if done:
                break

if __name__ == "__main__":
    run(trials, run_demos=run_demos)
    evaluate_policy()
   