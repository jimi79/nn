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
		self.nn.display_every_n_steps=1000 # we check cost functione very 1000 steps, but that doesn't apply here anyway, because we won't use train
		self.nn.verbose=False

		#self.last_res=None # last result i've got, so i can run BP without having to redo FP. But that is not very costy so it may not be necessary # wont store it for now


# define the max state value, the max action value.
# for the 0-9 game, max state is 100 (val from 0 to 99)
# and max action is from 1 to 10
# each is a bool 1 or 1
# new state outputting of the NN is the same size of the in value, so in is 110, out is 100.


	def pick_action(state, list_actions):
		# we try each actions against the NN, and then pick the one that lead to the status giving the more points

		a=np.zeros(list_actions)

		outputs=[] # possible outputs for each action
		points=[] # points available for a given action
		for i in range(list_actions):
			b=a
			b[i]=1
			input_=np.concat(state, b)
			res=np.FPdl(input_)
			outputs.append(res[-1].a) # it is a matrice here
			points.append(0) # i don't know yet 
		#so now i've got each output, and from there will be able to decide which is the best one. if max=0, then i pick at random. 
# i should create a list sorted by that. How to do that ? Well, max=0. list=[]. if val >max: max=val, list=[]. else list.append(action[i])
# i need an object just for that. or maybe i could do with 3 lists, but that is the same.
		max_points=0
		res=[]
		for i in range(list_actions):
			if points[i] > max_points:
				res=[]
			res.append(i) # that action is amongst the best outcome possible 
		a=random.range(len(res))
		return a[i] # index of the action that has the best outcome


	def learn(oldstate, action, newstate): # will have to concatenate oldstate and action, that are an array of booleans. Not sure how to do that though. Plus the action list may change from time to time.
# so action should be an integer, and i should adapt the size so it can handle the max action number
# and action here is the max value
