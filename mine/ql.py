#!/usr/bin/python3

import importlib
import nn
import numpy as np
import random
import utils_ql

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
		self.nn.display_every_n_steps=1000 # we check cost functione very 1000 steps, but that doesn't apply here anyway, because we won't use train
		self.nn.verbose=False
		self.verbose=False
		self.nn.nn.filename='nn09.tmp'

		#self.last_res=None # last result i've got, so i can run BP without having to redo FP. But that is not very costy so it may not be necessary # wont store it for now


# define the max state value, the max action value.
# for the 0-9 game, max state is 100 (val from 0 to 99)
# and max action is from 1 to 10
# each is a bool 1 or 1
# new state outputting of the NN is the same size of the in value, so in is 110, out is 100.



# at some point i will need to define an array of states, because i can't use matrices as index. Or i could just convert that into a binary, because it's always 0 and 1 so far. But that might change..... That would be better if i start with indices, but that would make the thing slower maybe

	def pick_action(self, state, list_actions):
		# we try each actions against the NN, and then pick the one that lead to the status giving the more points

		a=np.zeros(list_actions) 
		outputs=[] # possible outputs for each action
		points=[] # points available for a given action
		for i in range(list_actions):
			b=a
			b[i]=1
			input_=np.concatenate([state, b]) # input should be a line
			input_=input_
			res=self.nn.FP(input_)
			res=res>=0.5
			outputs.append(res) # it is a matrice here
			if self.verbose:
				print("FP with %d and action %d will be %d" % (temp_format(state), i, temp_format(res)))
			points.append(0) # i don't know yet 
		#so now i've got each output, and from there will be able to decide which is the best one. if max=0, then i pick at random. 
# i should create a list sorted by that. How to do that ? Well, max=0. list=[]. if val >max: max=val, list=[]. else list.append(action[i])
# i need an object just for that. or maybe i could do with 3 lists, but that is the same.
		max_points=0
		res=[]

		# here i need to random from time to time, in case i've got outcomes that are still at 0

		for i in range(list_actions): 
			if points[i] > max_points: #here : from time to time, if an outcomme is None (or 0 ?), then it will be considered as good enough, so that the computer doesn't stuck to a winning position if there are multiples path.
															# or maybe i should just lower all my values in my q learning array from time to time to force it to reevaluate some positions. Or randomize that array. I've got to think about it, that looks again like NN
				res=[]
			res.append(i) # that action is amongst the best outcome possible 
		i=res[random.randrange(len(res))] # i need to print that out with points so i can see what is the best outcome
		if self.verbose:
			print("I think outcome will be %d" % temp_format(outputs[i]))
		return i

	def learn(self, oldstate, list_actions, action, newstate): # will have to concatenate oldstate and action, that are an array of booleans. Not sure how to do that though. Plus the action list may change from time to time.
# so action should be an integer, and i should adapt the size so it can handle the max action number
# and action here is the max value
		a=np.zeros(list_actions) 
		a[action]=1
		input_=np.concatenate([oldstate, a]) # input should be a line

		if self.verbose:
			print("BP with %s" % temp_format2(input_))
		res=self.nn.FPdl(input_)
		self.nn.BP(newstate) # nn object has still in memory the datalayer, so it remembered what the output was
		

