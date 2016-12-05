#!/usr/bin/python3

import importlib
import nn
import numpy as np
import random
import utils_ql


def array_to_string(array):
	return utils_ql.temp_format(array)

class Qlearning():
	def __init__(self, max_state, max_action):
# max_action is all state included. it's just to prepare a input for the NN large enough
		self.max_state=max_state
		self.max_action=max_action 
		self.min_data_to_train=1000 # default value
		self.max_data_to_train=5000 # default value
		self.min_cpt_since_last_train=100 # don't train every time
		self.cpt_since_last_train=0 
		self.array_points={}
		self.alpha=0.8 # factor with which i should use the max

# nn setup
		self.nn=nn.Train()
		self.nn.check_every_n_steps=1000 # we check cost functione very 1000 steps, but that doesn't apply here anyway, because we won't use train
		self.nn.save_every_n_steps=-1
		self.nn.nn.verbose=False
		self.nn.min_J_cv=0.01
		self.nn.max_cpt=10000 
		self.nn.nn.filename='nn09.tmp' 
		self.verbose=False
		self.X=[]
		self.y=[]


		#self.last_res=None # last result i've got, so i can run BP without having to redo FP. But that is not very costy so it may not be necessary # wont store it for now


# define the max state value, the max action value.
# for the 0-9 game, max state is 100 (val from 0 to 99)
# and max action is from 1 to 10
# each is a bool 1 or 1
# new state outputting of the NN is the same size of the in value, so in is 110, out is 100.



# at some point i will need to define an array of states, because i can't use matrices as index. Or i could just convert that into a binary, because it's always 0 and 1 so far. But that might change..... That would be better if i start with indices, but that would make the thing slower maybe

	def pick_action(self, state, list_actions):
		# we try each actions against the NN, and then pick the one that lead to the status giving the more points

		outputs=[] # possible outputs for each action
		points=[] # points available for a given action

		state_str=array_to_string(state)

		for i in list_actions: 
			a=np.zeros(10)
			a[i]=1 
			input_=np.concatenate([state, a]) # input should be a line
			res=self.nn.FP(input_)
			output=res>=0.5
			outputs.append(output) # it is a matrice here 
			new_state_str=array_to_string(output) 
			b=self.array_points.get(new_state_str)
			points.append(b)
			if self.verbose:
				if b is None:
					b="?"
				else:
					b=str(b)
				print("%d+%d=%d(%s points)," % (utils_ql.temp_format(state),i+1,utils_ql.temp_format(output),b),end="") 

		if self.verbose:
			print("")
		max_points=None
		best_actions=[] 
		unknown_actions=[]
		avg=0 # average outcome of the status to come
		sum_=0
		cpt=0 
		for i in list_actions: 
			p=points[i]
			if not(p is None):
				sum_+=p
				cpt+=1 
			if not(p is None):
				if max_points is None:
					max_points=p
				if p > max_points: #here : from time to time, if an outcomme is None (or 0 ?), then it will be considered as good enough, so that the computer doesn't stuck to a winning position if there are multiples path.
																# or maybe i should just lower all my values in my q learning array from time to time to force it to reevaluate some positions. Or randomize that array. I've got to think about it, that looks again like NN
					best_actions=[]
				best_actions.append(i) # that action is amongst the best outcome possible 
			else:
				unknown_actions.append(i)

		if cpt>0:
			avg=sum_/cpt
			self.array_points[array_to_string(state)]=self.alpha*avg

		if len(best_actions)==0:
			best_action=random.choice(list_actions)
			if self.verbose:
				print("I picked an action at random, i have no idea")
		else:
			if max_points>0: 
				best_action=random.choice(best_actions)
				if self.verbose:
					print("I picked action %d because it is worth %d points" % (best_action, max_points))
			else:
				if len(unknown_actions)==0:
					best_action=random.choice(list_actions)
					if self.verbose:
						print("I picked an action that will make me loose, because i have no choice")
				else:
					best_action=random.choice(unknown_actions)
					if self.verbose:
						print("I picked action %d because that is one amongst which i won't loose" % (best_action))

		return best_action



	def learn(self, oldstate, list_actions, action, newstate, points=None): # will have to concatenate oldstate and action, that are an array of booleans. Not sure how to do that though. Plus the action list may change from time to time.
# so action should be an integer, and i should adapt the size so it can handle the max action number
# and action here is the max value
		a=np.zeros(list_actions) 
		a[action]=1
		input_=np.concatenate([oldstate, a]) # input should be a line
		output=newstate

		#if self.verbose:
		#	print("FP after %s" % utils_ql.temp_format(output))
		#res=self.nn.FPdl(input_)

		self.X.append(input_)
		self.y.append(output)

		while len(self.X) > self.max_data_to_train:
			self.X.pop(0)
			self.y.pop(0) # X and y have to be synchronized, no matter what

		self.cpt_since_last_train+=1
		if len(self.X) >= self.min_data_to_train:
			if self.cpt_since_last_train >= self.min_cpt_since_last_train:
				self.nn.datas.raw.X=np.array(self.X)
				self.nn.datas.raw.y=np.array(self.y)
				self.nn.datas.split() # split half
				self.nn.train()
				self.cpt_since_last_train=0
				self.nn.save()
		
		# got to handle points here.
		if points!=None:
			state=array_to_string(newstate)
			if self.verbose:
				print("I've got to remember that the state %d is worth %d" % (utils_ql.temp_format(newstate), points))
			self.array_points[state]=points 


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
