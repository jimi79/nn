#!/usr/bin/python3

import nn


class State:
	def __init__(self, ident, points=None):
		self.ident=ident
		if points==None:
			points=0 #not good, not bad
		self.points=points

class Qlearning():
	def __init__(self, max_state, max_action):
# max_action is all state included. it's just to prepare a input for the NN large enough
		self.max_state=max_state
		self.max_action=max_action 
		self.nn=nn.Train()
		self.nn.init_syns([self.max_state + self.max_action,self.max_state]) # one hidden layer
		self.nn.display_every_n_steps=None


# define the max state value, the max action value.
# for the 0-9 game, max state is 100 (val from 0 to 99)
# and max action is from 1 to 10
# each is a bool 1 or 1
# new state outputting of the NN is the same size of the in value, so in is 110, out is 100.


	def pick_action(state, list_actions):
		# we try each actions against the NN, and then pick the one that lead to the status giving the more points

		a=np.zeros(list_actions)
		output={}
		for i in range(list_actions):
			b=a
			b[i]=1
			input_=np.concat(state, b)
			output[i]=np.FP(input_)
# and we ask the NN what can goes out with that





	def learn(oldstate, action, newstate): # will have to concatenate oldstate and action, that are an array of booleans. Not sure how to do that though. Plus the action list may change from time to time.
# so action should be an integer, and i should adapt the size so it can handle the max action number
# and action here is the max value
