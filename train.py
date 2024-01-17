"""
Julia Santaniello
Started: 06/06/23
Last Updated: 12/21/23

For training a single vanilla policy. Input hyperparamters and saved policy name.
"""
import numpy as np
import time
from PIL import Image
import pandas as pd
import csv
import pygame
from algorithms.simplePG import SimplePG
from games.lunar_lander import LunarLander
from imports import environment, demo_name, seed, steps, gamma, exploration_type, greedy_e_epsilon, learning_rate, obs_size_values, num_actions,algorithm_name, env_name, episodes, trials


env = environment()
demo_name="train"
"Train an agent without demonstration"

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
def run_train(trials):
    # start()
    # demo.main(demo_name)
    print("Thank you!")
    time.sleep(1)
    print("Starting agent training...")

    r,s,a=[],[],[]
    for i in range(trials):
        rewards, steps, accuracy = train()
        r.append(rewards)
        s.append(steps)
        a.append(accuracy)
    
    r = np.asarray(r).mean(axis=0)
    s = np.asarray(s).mean(axis=0)
    a = np.asarray(a).mean(axis=0)

    print("Done! Saving Data!")
    save_data(r, s, a)

"""
trains the agent with PG algorithm and preferred number of demonstrations
"""
def train():
    agent = SimplePG(num_actions=num_actions, input_size=obs_size_values, hidden_layer_size=12, learning_rate=learning_rate, decay_rate=0.99, gamma=gamma, greedy_e_epsilon=greedy_e_epsilon, random_seed=10)
    
    avg_rewards_past, avg_steps_past, avg_accuracy_past = [],[],[]
    state = env.reset(seed=10)
    sum_wins, running_reward, running_steps = 0,0,0


    for i_episode in range(0, episodes):
        state = env.reset()
        episode_rewards = 0
        next_state=None

        for t in range(steps):
            action = agent.pickAction(state, exploring=True, exploration_type="decay")
            next_state, reward, done, win = env.step(action)

            if t == steps -1:
                reward = -100

            episode_rewards += reward
            agent.saveStep(state=state, reward=reward, action=action, next_state=next_state, done=done)
            
            state = next_state

            if done:
                running_steps += t
                running_reward += episode_rewards
                if win:
                    sum_wins+=1
                break

        if t == steps -1:
            running_reward -= 100
        
        # Updating the policy :
        agent.finishEpisode()

        if exploration_type == "decay":
            agent._explore_eps -= 1/5000

        if i_episode % 20 == 0:
            avg_accuracy_past.append(sum_wins/20)
            avg_rewards_past.append(running_reward/20)
            avg_steps_past.append(running_steps/20)
            print('Episode {}\tlength: {}\treward: {}\t accuracy: {}'.format(i_episode, t, running_reward/20, sum_wins/20))
            if sum_wins/20 > 0.8:
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

        for j in range(steps):
            action = agent.pickAction(state, exploring=False)

            #Return state, reward
            new_state, reward, done, _ = env.step(action)

            state = new_state
            if done:
                break
"""
Saves training data to csv file
"""
def save_data(r,s,a):
    arr = np.asarray([r,s,a])
    df = pd.DataFrame(arr)
    df.to_csv("./data/results/{}.csv".format(demo_name))

if __name__ == "__main__":
    run_train(trials)


#random_seed = 543
#torch.manual_seed(random_seed)
#env = gym.make('LunarLander-v2')
#env.seed(random_seed)
    
# if average_reward > 210:
#     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
#     print("########## Solved! ##########")
#     #test(name='LunarLander_{}}.pth')#.format(title))
#     break