import time
import pygame
from algorithms.simplePG import SimplePG
import pickle
from imports import environment, demo_name_hitl, seed, steps, gamma, learning_rate, obs_size_values, num_actions,algorithm_name, env_name, episodes, num_demos

"""
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

Saves demonstrations from human expert.
"""

policy = SimplePG(num_actions=num_actions, input_size=obs_size_values, hidden_layer_size=12, learning_rate=learning_rate, decay_rate=0.99, gamma=gamma, greedy_e_epsilon=0.1, random_seed=seed)
env = environment(render_mode="human")
env.render()
demo_name = demo_name_hitl

"""
Human can play the game in real time using these keys
"""
def human_play():
    pressed_keys = pygame.key.get_pressed()
    if env_name == "lunar lander":
        if pressed_keys[pygame.K_LEFT]: #left
            return 1
        elif pressed_keys[pygame.K_UP]: #up
            return 2
        elif pressed_keys[pygame.K_RIGHT]: #right
            return 3
        return 0 #do nothing
    elif env_name == "snake":
        if pressed_keys[pygame.K_LEFT]: #left
            return 0
        elif pressed_keys[pygame.K_UP]: #up
            return 2
        elif pressed_keys[pygame.K_RIGHT]: #right
            return 1
        elif pressed_keys[pygame.K_DOWN]:
            return 3
        else:
            return None
    elif env_name == "car_racing":
        if pressed_keys[pygame.K_RIGHT]:
            return 1
        if pressed_keys[pygame.K_LEFT]:
            return 2
        if pressed_keys[pygame.K_UP]:
            return 3
        if pressed_keys[pygame.K_DOWN]:
            return 4
        else: return 0
        


"""
Only saves dmonstrations
"""
def demonstrations_only():
    demonstrations_dict = {"demo_name": demo_name, "algorithm": algorithm_name} #dictionary of demonstrations
    
    #Rewards per epsiode saved
    rewards_per_episode = []
    seed_ = seed

    #for each demonstration desired
    for i_episode in range(0, num_demos):
        time.sleep(1)
        seed_ += i_episode
        timestamps, action_list = [], [] #timestamps
        running_reward = 0
        state = env.reset(seed=seed_) #reset
        human_action = 0
        print(i_episode)
        env.render()

        for t in range(steps):
            #pick an action
            #pygame.event.wait(timeout=500)
            action = human_play()

            if action is not None:
                human_action = action
            
            action_list.append(human_action)
            next_state, reward, done, win = env.step(human_action)
            env.render()
            running_reward += reward

            policy.saveHumanStep(state=state, reward=reward, action=human_action)
            
            state = next_state

            #timestamp update
            current_time = time.time()
            timestamps.append(current_time)

            #Save action
            policy.save_human_action(human_action)

            running_reward += reward

            if done:
                break
        
        # Updating the policy :

        final_episode_actions = action_list
        final_episode_rewards = policy.h_rewards
        final_episode_states = policy.h_states
        final_episode_steps = t


        assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards) == len(final_episode_states))
        rewards_per_episode.append(running_reward)
        finished_demo = save_demonstration(env_name, final_episode_steps, final_episode_states, timestamps, final_episode_actions, final_episode_rewards, seed=seed_, environment_version=1.0)
        demonstrations_dict[i_episode] = finished_demo
        policy.resetLists()

    return demonstrations_dict

"""
Keeps demonstration in a python dictionary
"""
def save_demonstration(environment_name, steps, states, timestamps, actions, rewards, seed, environment_version):
    
    assert(len(rewards) == len(states) == len(timestamps) == len(actions))

    state_dict = {}

    state_dict["environment_name"] = environment_name
    state_dict["environment_version"] = environment_version
    state_dict["seed"] = seed
    state_dict["steps"] = steps
    state_dict["timestamps"] = timestamps
    state_dict["states"] = states
    state_dict["actions"] = actions
    state_dict["rewards"] = rewards

    return state_dict

"""
Can watch lunar lander play by itself through the saved demonstration
"""
def play_demonstrations(demo_name_):
    file = open('./data/demonstrations/{}.pickle'.format(demo_name_), 'rb')
    demo_dict = pickle.load(file)
    file.close()
    env = environment()
    env.render()

    for i in range(len(demo_dict)-2):
        steps = demo_dict[i]["steps"]
        seed_ = demo_dict[i]["seed"]
        state = env.reset(seed=seed_)

        for j in range(steps):
            action = demo_dict[i]["actions"][j]

            #Return state, reward
            state, reward, done, _ = env.step(action)

            if done:
                break

"""
Save to pickle file in Data/Demonstration folder
"""
def save_demo(demo, demo_name_):
    with open('./data/demonstrations/{}.pickle'.format(demo_name_), 'wb') as handle:
            pickle.dump(demo, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
Runs and saves demonstrations
"""
def main(demo_name_):
    demo = demonstrations_only()
    #save_demo(demo, demo_name_)

if __name__=="__main__":
    main(demo_name)
    #play_demonstrations(demo_name)