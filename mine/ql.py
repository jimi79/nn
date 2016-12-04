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

		if self.datasets.raw.X is None:
			self.datasets.raw.X=[input_]
		else: 
			self.datasets.raw.X.append([input_])




		self.nn.BP(newstate) # nn object has still in memory the datalayer, so it remembered what the output was
		
# ok, so now, is the FP/BP method any good, or do i need to build a package ?

# it seems i need to build a big package of restuls though.
# meaning that, i will do that, have a max size (taht can be -1), and learn each time that package reach a min value.

# thinking ktning
# variables

# number of min lines in a dataset before learning
# should i learn each time that big dataset ?
# should i learn a big dataset, then make adjustments ?
# should i just remove doublons from the dataset (how ?) and learn over and over that one ?
# should i do a BP till Jcv drops, from time to time, to correctly assimilate the new value ?
# and then i will maybe someday implement the actual QL


# use the loop, over 1000 exemples. check the nn that is trained on the way, and compare perf with the nn that is trained on the whole dataset.
# that is not fair, the nn trained over the whole dataset is trained n times over that 1000 examples dataset.
# compare a loop over 10000 with an AI that updates each time
# with an AI that has a 1000 example dataset, and train 10 times max
# why doing that : well, check that an AI trained nnnn times for each value is at least as good as an AI trained on the whole dataset. Because my question is : is an AI that does BP for only one value can destroy all the rest ? (i think it can)

# answer, or at least one of them : https://www.researchgate.net/post/Which_one_is_better_between_online_and_offline_trained_neural_network -> yes, batch is better. meaning that i should play 1000 times, then study the batch.
# so the learn function here should be able to put aside stuff every n times, and then use the actual train function



# modify Learn function to actually feed the dataset, and if its size > 1000, then remove stuff in double (X and y should match though, no idea how i will do that), or i remove only X that are in double, taking the most recent (highest index), on the basis that rtahrqwkejrhkjqlwherkjqwehlrj
# ok that is wrong. having multiples values can help under some circumstances
# otherwise i don't need a NN, because i'm assuming the outcome is definitive

# SO : we wait till we have n datas when asked to 'learn'. when we got them, we learn (train) till Jcv or Jtrain drops under a given value (Jcv is better). split will left 0 for test. when we got over m datas, we remove n of them.
# and in learn, we try to do qlearning too. value of status is max of leading status. For that i think i need an array, that will use the integer version of the bool of status. Or i use the AI to find out what status is the next to the current status i'm evaluating. I think using AI is good because that is what i know how to reach.

# + test alpha with the classic nn test to see if it learns faster or not
# test cost function on cv and train before training, in case the network is already ok
