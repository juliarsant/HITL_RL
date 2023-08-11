import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_elements:int, human:bool):
        super(ActorCritic, self).__init__()

        #linear transformation (state)
        self.state_layer = nn.Linear(state_elements, 128)

        #policy actions (used in actor)
        self.action_layer = nn.Linear(128, 4)

        #Evaluation of policy actions (used in critic)
        self.value_layer = nn.Linear(128, 1)
        
        #HITL
        self.human = human
        self.hitl_rewards = []
        self.hitl_state_values = []
        self.hitl_probsbuff = []

        #buffers
        self.probsbuff = []
        self.state_values = []
        self.rewards = []

    def forward(self, state, argmax):
        state = torch.from_numpy(state).float()

        #Activation
        state = F.relu(self.state_layer(state))

        #actor: chooses an action to take from the state
        action_probs = F.softmax(self.action_layer(state), dim = 0)

        #Critic = estimated value for being in this state
        state_value = self.value_layer(state)

        #Creates a categorial distribution from the array of action probs
        policy_distribution = Categorical(action_probs)

        #samples an action based on this distribution
        if argmax:
            action = np.argmax(action_probs.detach().numpy())
            action = np.array(action)
            action = torch.from_numpy(action)
        else:
            action = policy_distribution.sample()
        
        #buffer gets chosen action and state values
        self.probsbuff.append(policy_distribution.log_prob(action))
        self.state_values.append(state_value)
        
        #returns item from a tensor with one element
        return action.item()
    
    def calculateLoss(self, gamma):
        
        # calculating discounted rewards:
        # will be in backwards order which the rewards were calculated 
        rewards = []
        discounted_rewards = 0
        hitl_rewards = []
        hitl_discounted_rewards = 0

        for reward, hitl_reward in zip(self.rewards[::-1], self.hitl_rewards[::-1]): #reversed rewards
            discounted_rewards = reward + gamma * discounted_rewards
            hitl_discounted_rewards = hitl_reward + gamma * hitl_discounted_rewards

            rewards.insert(0, discounted_rewards)
            hitl_rewards.insert(0, hitl_discounted_rewards)

                
        # normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std())

        hitl_rewards = torch.tensor(hitl_rewards)
        hitl_rewards = (hitl_rewards - hitl_rewards.mean()) / (hitl_rewards.std())

        loss = 0
        for prob, value, discounted_rewards in zip(self.probsbuff, self.state_values, rewards):
            advantage = (discounted_rewards  - value.item())

            #actor loss
            action_loss = -prob * advantage

            #critic loss
            value_loss = F.smooth_l1_loss(value[0], discounted_rewards)


            loss += (action_loss + value_loss) 

        return loss
    
    def clearMemory(self):
        del self.probsbuff[:]
        del self.state_values[:]
        del self.rewards[:]
