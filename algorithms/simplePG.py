import numpy as np
import time


"""
Julia Santaniello
Started: 06/01/23
Last Updated: 12/21/23

SimplePG: Vanilla Policy Gradient Algorithm that uses a simple neural network to 
		  train agent to play a game based on state observations.
"""

class SimplePG(object):
	def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate,gamma,decay_rate,greedy_e_epsilon,random_seed):
		# store hyper-params
		self.num_actions = num_actions
		self.input_size = input_size
		self.hidden_layer_size = hidden_layer_size
		self._learning_rate = learning_rate
		self._decay_rate = decay_rate
		self._gamma = gamma
		self.random_seed = random_seed
		
		# some temp variables
		self.past_states, self.past_hidden_states, self._dlogps, self.past_rewards = [],[],[],[]

		# some hitl temp variables
		self.h_actions, self.h_rewards, self.h_wins, self.h_states = [], [], [], []
		self.a_probs = []

		# variables governing exploration
		self._exploration = True # should be set to false when evaluating
		self._explore_eps = greedy_e_epsilon
		
		#create model
		self.init_model(self.random_seed)
		
	
	def init_model(self,random_seed):
		# create model
		#with cp.cuda.Device(0):
		self._model = {}
		np.random.seed(random_seed)
	   
		# weights from input to hidden layer   
		self._model['W1'] = np.random.randn(self.input_size,self.hidden_layer_size) / np.sqrt(self.input_size) # "Xavier" initialization
	   
		# weights from hidden to output (action) layer
		self._model['W2'] = np.random.randn(self.hidden_layer_size,self.num_actions) / np.sqrt(self.hidden_layer_size)
			
				
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory

	def save_agent(self):
		print("skipped save agent")
		pass
	
	# softmax function
	def softmax(self,x):
		probs = np.exp(x - np.max(x, axis=1, keepdims=True))
		probs /= np.sum(probs, axis=1, keepdims=True)
		return probs
		
	def discount_rewards(self,r):
		""" take 1D float array of rewards and compute discounted reward """
		discounted_r = np.zeros_like(r)
		running_add = 0
		for t in reversed(range(0, r.size)):
			running_add = running_add * self._gamma + r[t]
			discounted_r[t] = float(running_add)
    
		return discounted_r
	
	# feed input (x) to network and get result
	def policy_forward(self,x):
		if(len(x.shape)==1):
			x = x[np.newaxis,...]

		h = x.dot(self._model['W1'])
		
		if np.isnan(np.sum(self._model['W1'])):
			print("W1 sum is nan")
		
		if np.isnan(np.sum(self._model['W2'])):
			print("W2 sum is nan")
		
		if np.isnan(np.sum(h)):
			print("nan")
			
			h[np.isnan(h)] = np.random.random_sample()
			h[np.isinf(h)] = np.random.random_sample()
			

		if np.isnan(np.sum(h)):
			print("Still nan!")
		
		
		# h[h<0] = 0 # ReLU nonlinearity
		# print("h: ", h)
		logp = h.dot(self._model['W2'])
		# print(logp)
		logp *= 0.3

		min = np.min(logp)
		logp += abs(min)
		max = np.max(logp)

		logp /= max

		# p = self.softmax(logp)
		# return p, h
  
		return logp, h # return probability of taking actions, and hidden state
		
	
	def policy_backward(self, hs_stack, epdlogp):
		""" backward pass. (eph is array of intermediate episode hidden states) 
			epdlogp is the """
		dW2 = hs_stack.T.dot(epdlogp)  
		dh = epdlogp.dot(self._model['W2'].T)
		# dh[hs_stack <= 0] = 0 # backpro prelu
  
		t = time.time()
  
		# if(be == "gpu"):
		#   self._dh_gpu = cuda.to_gpu(dh, device=0)
		#   self._epx_gpu = cuda.to_gpu(self._epx.T, device=0)
		#   self._dW1 = cuda.to_cpu(self._epx_gpu.dot(self._dh_gpu) )
		# else:
		
		self._dW1 = self._epx.T.dot(dh) 
    
		return {'W1':self._dW1, 'W2':dW2}
	
	def set_explore_epsilon(self,e):
		self._explore_eps = e
	
	# input: current state/observation
	# output: action index
	def pickAction(self, state_obs, exploring, exploration_type=None):
		final_action = None
		# feed input (x) through network and get output: action probability distribution and hidden layer
		aprob, h = self.policy_forward(state_obs)
		a2 = -1
		# if exploring
		if exploring:
			# greedy-e exploration
			rand_e = np.random.uniform()
			if rand_e < self._explore_eps:
				a2 = np.random.randint(0,4)
				# set all actions to be equal probability
				aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]
		# else:
		# 	final_action = np.argmax(aprob)


		if np.isnan(np.sum(aprob)):
			aprob[0] = [ 1.0/len(aprob[0]) for i in range(len(aprob[0]))]

		aprob_cum = np.cumsum(aprob)
		u = np.random.uniform()
		a = np.where(u <= aprob_cum)[0][0]	

		# record various intermediates (needed later for backprop)
		self.past_states.append(state_obs) # observation
		self.past_hidden_states.append(h)

		#Probabilities of actions
		self.a_probs.append(aprob)

		#softmax loss gradient
		# dlogsoftmax = aprob.copy()
		dlogsoftmax = self.softmax(aprob.copy())

		# dlogsoftmax[0,a] -= 1 #-discounted reward WRONG
		self._dlogps.append(dlogsoftmax)

		if a2 > -1:
			a = a2

		if final_action is not None:
			return final_action
		
		return a
		
	# after process_step, this function needs to be called to set the reward
	def saveStep(self,reward,state=None, action=None, next_state=None, done=None):
		# store the reward in the list of rewards
		self.past_rewards.append(reward)
		
	# reset to be used when evaluating
	def reset(self):
		self.past_states,self.past_hidden_states,self._dlogps,self.past_rewards, self.h_actions, self.h_states = [],[],[],[],[],[] # reset 
		self._grad_buffer = { k : np.zeros_like(v) for k,v in self._model.items() } # update buffers that add up gradients over a batch
		self._rmsprop_cache = { k : np.zeros_like(v) for k,v in self._model.items() } # rmsprop memory
		
	def resetLists(self):
		self.past_states,self.past_hidden_states,self._dlogps,self.past_rewards, self.h_rewards, self.h_states = [],[],[],[],[],[] # reset 


	# this function should be called when an episode (i.e., a game) has finished
	def finishEpisode(self, ep_return = None, human_demonstration=False):
		# stack together all inputs, hidden states, action gradients, and rewards for this episode
		
        #stack of past states
		self._epx = np.vstack(self.past_states)
		
		#stack of past hidden states
		hs_stack = np.vstack(self.past_hidden_states)
		
		
		#stack of the softmax probabilities over actions selected by the agent
		epdlogp = np.vstack(self._dlogps)

		# stack of agent action probabilities
		agent_actions = np.vstack(self.a_probs)
		
		# print("a:",agent_actions)
		# print("e: ", epdlogp)
		# self.past_rewards is the stack of rewards from the episode
		epr = np.vstack(self.past_rewards)

		if human_demonstration:
			human_actions = np.vstack(self.h_actions)
			human_rewards = np.vstack(self.h_rewards)
		
		self.past_states,self.past_hidden_states,self._dlogps,self.past_rewards, self.h_actions, self.a_probs, self.h_rewards = [],[],[],[],[],[],[] # reset array memory

		#compute the discounted reward backwards through time
		discounted_epr = (self.discount_rewards(epr))

		#Mean of the discounted rewards
		discounted_epr_mean = np.mean(discounted_epr)
		
		# standardize the rewards to be unit normal (helps control the gradient estimator variance)
		discounted_epr_diff = np.subtract(discounted_epr,discounted_epr_mean)

		#Variance
		discounted_epr_diff /= np.std(discounted_epr)

		# print("epr: ", epr)
		# print("dicount epr: ", discounted_epr_diff)
		# print("epdlog: ", epdlogp)
		
		if not human_demonstration:
			epdlogp *= discounted_epr_diff # modulate the gradient with advantage (PG magic happens right here.)
		else:
			scores = self.human_robot_agreement_score(epdlogp, human_actions, 4)
			discounted_score = self.discount_rewards(scores)
			# discounted_score = self.discount_rewards(human_rewards)
			discounted_score_mean = np.mean(discounted_score)
			discounted_score_diff = np.subtract(discounted_score,discounted_score_mean)
			discounted_score_diff /= np.std(discounted_score)

			# print("score: ", discounted_score_diff)


			epdlogp *=  discounted_score_diff # modulate the gradient with advantage (PG magic happens right here.)

        #gradient over batch
		grad = self.policy_backward(hs_stack, epdlogp)
		
		for k in self._model: self._grad_buffer[k] += grad[k] # accumulate grad over batch

	def saveHumanStep(self, reward, state=None, action=None):
		self.h_rewards.append(reward)
		self.h_states.append(state)

	#Saves an array that represents the human demonstration "distribution"
	def save_human_action(self, human_action):
		x = np.zeros(self.num_actions)
		x[human_action] = 1.0
		
		self.h_actions.append(x)

	def human_robot_agreement_score(self, a_robot, a_human, num_actions):
		score = 0
		score_list = []

		for i in range(len(a_robot)):
			score = 0
			# print("human: ", a_human[i])
			# print("robot: ", a_robot[i])
			for j in range(num_actions):

				if a_human[i][j] == 1.0:
					score += a_robot[i][j]
				else:
					score -= a_robot[i][j]
			score_list.append(score)

		return np.vstack(score_list)


	# called to update model parameters, generally every N episodes/games for some N
	def updateParamters(self):
		for k,v in self._model.items():
			g = self._grad_buffer[k] # gradient
			self._rmsprop_cache[k] = self._decay_rate * self._rmsprop_cache[k] + (1 - self._decay_rate) * g**2
			self._model[k] -= self._learning_rate * g / (np.sqrt(self._rmsprop_cache[k]) + 1e-5)
			self._grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer

		
		
# #Make one hot encoding vector of actions from human
# #1 if perfect agree, -1 if perfectly disagree, other if prob is 1 and 0.76 for example
# # take discounted rewards, normalize with mean and std
# #make a separate function for human and autonomous finish_epsiode() types

# """
# J:
# Create a one hot encoding of the actions from the user. 
# Use the equation jivko gave on the whiteboard to create the advantage
# Get the discounted rewards for the USER, normalize it with the mean and std
# Do I use the discounted reward to compute the advantage or the equation jivko gave?
# multiply the (autonomous) probabilities by the advantage
# """
