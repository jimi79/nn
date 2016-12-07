#!/usr/bin/python3

import importlib
import nn
import numpy as np
import random
import os
import pickle



# CHANGELOG
# v2: it will store the history itself, so when u tell it to learn, it will based on what it has in memory (and the new status)

def array_to_integer(array):
	return sum([array[i]*(2**i) for i in range(len(array))])

def integer_to_array(integer):
	a=[int(x) for x in bin(integer)[2:]]	
	a.reverse()
	while len(a)<18:
		a.append(0)
	return a


class Qlearning():
	def __init__(self, max_state, max_action):
# max_action is all state included. it's just to prepare a input for the NN large enough
		self.max_state=max_state
		self.max_action=max_action 
		self.min_data_to_train=5000 # default value
		self.max_data_to_train=50000 # default value
		self.min_cpt_since_last_train=1000 # don't train every time
		self.cpt_since_last_train=0 
		self.array_points={}
		self.alpha=0.8 # factor with which i should use the max
		self.filename=None
		self.logfilename=None



# nn setup
		self.nn=nn.Train()
		self.nn.check_every_n_steps=1000 # we check cost functione very 1000 steps, but that doesn't apply here anyway, because we won't use train
		self.nn.save_every_n_steps=1000
		self.nn.nn.verbose=False
		self.nn.min_J_cv=0.01
		self.nn.max_cpt=1000 
		self.nn.nn.filename='nn09.tmp' 
		self.nn.nn.alpha=1
		self.nn.nn.lambda_=3
		self.verbose=False
		self.X=[]
		self.y=[]

	def append_log(self, text):
		if not self.logfilename is None:
			with open(self.logfilename, "a") as f:
				f.write("%s\n" % text)



	def restart(self): 
		self.history_status=[] # will be feed by 'learn'
		self.history_actions=[] # will be feed by 'pick_action'

	def pick_action(self, state, list_actions):
		outputs={} # possible outputs for each action
		points={} # points available for a given action 
		s=""
		for i in list_actions: 
			a=np.zeros(self.max_action)
			a[i]=1 
			input_=np.concatenate([state, a]) # input should be a line
			res=self.nn.FP(input_)
			output=res>=0.5
			outputs[i]=output # it is a matrice here 
			new_state=array_to_integer(output)
			b=self.array_points.get(new_state)
			points[i]=b
			if b is None:
				b="?"
			else:
				b=str(b)
			s+="%d+%d=%d(%s points)," % (array_to_integer(state),i,new_state,b)
		if self.verbose:
			print(s)
		self.append_log(s)
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
				if max_points is None:
					max_points=p
					best_actions=[]
				if p > max_points:
					best_actions=[]
					max_points=p
				if p >= max_points:
					best_actions.append(i)
			else:
				unknown_actions.append(i) 

		# from time to time : encourage unknown paths
		if cpt>0:
			avg=sum_/cpt
			self.array_points[array_to_integer(state)]=self.alpha*max_points
			self.append_log("max outcome for state %d is %d, so it is worth %d" % (array_to_integer(state), max_points, self.array_points[array_to_integer(state)]))
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
		self.history_actions.append(best_action) 
		return best_action


	def learn(self, newstate, points=None): 
		if len(self.history_actions)>1:
			a=np.zeros(self.max_action) 
			a[self.history_actions[-1]]=1
			input_=np.concatenate([self.history_status[-1], a]) # input should be a line
			output=newstate 
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
					self.nn.datas.split_half() # split half
					self.nn.train()
					self.cpt_since_last_train=0
					#self.nn.save()
			
			# got to handle points here.
			if points!=None:
				state=array_to_integer(newstate)
				self.append_log("state %d is worth %d" % (array_to_integer(newstate), points))
				if self.verbose:
					print("I've got to remember that the state %d is worth %d" % (array_to_integer(newstate), points))
				self.array_points[state]=points 
		self.history_status.append(newstate) 

	def save(self):
		pickle.dump(self.array_points, open(self.filename, 'wb'))

	def try_load(self):
		if os.path.exists(self.filename):
			self.array_points=pickle.load(open(self.filename, 'rb'))
			return True
		else:
			return False
