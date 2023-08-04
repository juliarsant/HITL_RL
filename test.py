from model import ActorCritic
import torch
from PIL import Image
from lunar_lander import LunarLander
import numpy as np
import gym


def test(n_episodes=100, name='LunarLander_MODIFIED_JS1.pth'):
    env = LunarLander(render_mode = "human")
    policy = ActorCritic(11)
    num_wins = 0
    
    policy.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    policy.eval()
    render = False
    save_gif = False

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        win = False
        for t in range(1500):
            action = policy(state, False) #Might want true for testing
            state, reward, done, _, win = env.step(action)
            running_reward += reward
            if win:
                num_wins += 1
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            if done:
                break
        #print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    accuracy = (num_wins/n_episodes)*100
    env.close()

    print(f"Accuracy: {accuracy}")



def mid_training_test(policy, mod, argmax):
    if not mod:
        #random_seed = 543
        #seed = torch.manual_seed(random_seed)
        env = gym.make('LunarLander-v2')
        #env.seed(seed)
    elif mod:
        env = LunarLander()
    
    num_wins = 0
    n_episodes = 20
    render = False
    save_gif = False
    rewards = []

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        win = False
        episode_rewards = 0
        for t in range(1500):
            action = policy(state, argmax)
            state, reward, done, _, win = env.step(action)
            running_reward += reward
            episode_rewards += reward
            if mod and win:
                num_wins += 1
            if render:
                 env.render()
                 if save_gif:
                     img = env.render(mode = 'rgb_array')
                     img = Image.fromarray(img)
                     img.save('./gif/{}.jpg'.format(t))
            
            if done:
                break
        if not mod and episode_rewards > 200:
            num_wins += 1
        rewards.append(episode_rewards)
        #print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    mean_rewards = np.mean(rewards)
    std_rewards = np.std(rewards)
    accuracy = (num_wins/n_episodes)*100
    env.close()
    print(f"Accuracy: {accuracy}, Mean: {mean_rewards}, STD: {std_rewards}")
    return mean_rewards, std_rewards, accuracy

            
if __name__ == '__main__':
    test()
