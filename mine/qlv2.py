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

class NN():
	def __init__(self):
		self.cpt=0
		self.min_cpt=1000
		self.min_data=1000
		self.max_data=5000
		self.X=[]
		self.y=[]
		self.nn=nn.Train()
		self.nn.params.check_every_n_steps=1000 # we check cost functione very 1000 steps, but that doesn't apply here anyway, because we won't use train
		self.nn.params.save_every_n_steps=1000
		self.nn.params.verbose=False
		self.nn.params.min_J_cv=0.01
		self.nn.params.max_cpt=1000 
		self.nn.params.filename=None
		self.nn.params.alpha=1
		self.nn.params.lambda_=3 

	def add(self, X, y):
		self.X.append(X)
		self.y.append(y)
		if len(self.X) > self.max_data:
			self.X.pop(0)
			self.y.pop(0)
		self.cpt+=1
	
	def should_i_train(self):
		return (self.cpt>self.min_cpt) and (len(self.X)>self.min_data)

	def train_if_needed(self):
		if self.should_i_train():
			self.nn.datas.raw.X=np.array(self.X)
			self.nn.datas.raw.y=np.array(self.y)
			self.nn.datas.split_half() 
			self.nn.train()
			self.cpt=0 
			return True
		else:
			return False

class Qlearning():
	def __init__(self, max_state, max_action):
# max_action is all state included. it's just to prepare a input for the NN large enough
		self.array_wins=[]
		self.alpha=0.8 # factor with which i should use the max
		self.filename=None
		self.logfilename=None
		self.verbose=False 
		self.nn_action=NN()
		self.nn_opponent=NN()
		self.nn_action.min_J_cv=0.01 # easy to see what the outcome is from your own actions
		self.nn_opponent.min_J_cv=1.3 # harder to foresee what the opponent will play
		self.winlist={} # index = True if win
		self.points={} # index = points
		self.restart # init some other variables

	def set_name(self, name):
		self.name=name

	def set_verbose(self, verbose):
		self.verbose=verbose
		self.nn_action.nn.params.verbose=verbose
		self.nn_opponent.nn.params.verbose=verbose

	def append_log(self, text):
		if not self.logfilename is None:
			with open(self.logfilename, "a") as f:
				f.write("%s\n" % text)
		if self.verbose:
			print(text)

	def restart(self): 
		self.last_board=None
		self.last_action=None

	def pick_action(self, state, list_actions):
		self.last_board=state
		points={} # points available for a given action 
		s="" 
		n_winlist=[] # that action leads to a direct win # n_ like now
		n_willwin=[] # that action leads to a probable win
		n_maxpoint_pos=None # max points for a good action
		n_maxpoint_neg=None # max points for a bad action
		n_maxpoint=None
		n_dontknow=[] # 
		n_willlose=[] # that action leads to a possible loss
		for i in list_actions: 
			a=np.zeros(self.max_action)
			a[i]=1 
			input_=np.concatenate([state, a]) # input should be a line
			res=self.nn_action.nn.FP(input_)
			output=res>=0.5
			new_state=array_to_integer(output) 
			p=None
			if not self.winlist.get(new_state) is None:
				ps="win"
				n_winlist.append(i)
				p=1000
			else: 
				res_op=self.nn_opponent.nn.FP(output) 
				new_state=array_to_integer(res_op) 
				p=self.points.get(new_state)
				ps="?"
				if not p is None:
					ps=str(p)
					if p > 0:
						if n_maxpoint_pos is None:
							n_maxpoint_pos=p
						if p > n_maxpoint_pos: 
							n_maxpoint_pos=p
							n_willwin=[]
						n_willwin.append(i)
					else:
						if n_maxpoint_neg is None:
							n_maxpoint_neg=p
						if p > n_maxpoint_neg: 
							n_maxpoint_neg=p
							n_willlose=[]
						n_willlose.append(i)
				else:
					n_dontknow.append(i) 

			self.append_log("%d+%d=%d(%s points)" % (array_to_integer(state),i,new_state,ps)) # %s to handle None 
			if not p is None:
				if n_maxpoint is None:
					n_maxpoint=p
				else:
					if p > n_maxpoint:
						n_maxpoint=p

		if not n_maxpoint is None:
			val=self.alpha*n_maxpoint
			self.points[array_to_integer(state)]=val
			self.append_log("update state %d with %0.2f points" % (array_to_integer(state), val))

		action=None
		if len(n_winlist)!=0:
			action=random.choice(n_winlist)
			text="%s picks winning action %d" % (self.name, action)
		else:
			if len(n_willwin)!=0:
				action=random.choice(n_willwin)
				text="%s picks the best action %d for %d points" % (self.name, action, n_maxpoint_pos)
			else:
				if len(n_dontknow)!=0:
					action=random.choice(n_dontknow)
					text="i picks the unknown action %d" % (action)
				else:
					if len(n_willlose)!=0:
						action=random.choice(n_willlose)
						text="%s picks the less bad action %d, for %d points" % (self.name, action, n_maxpoint_neg)
		self.append_log(text)
		self.last_action=action
		return action

	def learn_action(self, newstate): 
		a=np.zeros(self.max_action) 
		a[self.last_action]=1
		input_=np.concatenate([self.last_board, a]) # input should be a line
		output=newstate 
		self.nn_action.add(input_, output) 
		if self.nn_action.train_if_needed():
			if self.verbose_training:
				if not self.name is None:
					print("%s training with nn_action, j_cv=%.04f" % (self.name,self.nn_action.nn.check(self.nn_action.nn.datas.cvset)))

		self.last_board=newstate
		self.append_log('I learn that %d with action %d leads to %d' % (array_to_integer(self.last_board), self.last_action, array_to_integer(newstate)))
	
	def learn_opponent(self, newstate): 
		if self.last_board != None:
			input_=self.last_board
			output=newstate 
			self.nn_opponent.add(input_, output) 
			if self.nn_opponent.train_if_needed(): 
				if self.verbose_training:
					if not self.name is None:
						print("%s training with nn_opponents, j_cv=%0.4f" % (self.name,self.nn_opponent.nn.check(self.nn_opponent.nn.datas.cvset)))
			self.append_log('I learn that %d and the opponent playing leads to %d' % (array_to_integer(self.last_board), array_to_integer(newstate)))
			return True
		else:
			return False

	def learn_points(self, state, points, win):
		state=array_to_integer(state)
		self.append_log("state %d is worth %d" % (state, points))
		if self.verbose:
			print("I've got to remember that the state %d is worth %d" % (state, points))
		self.points[state]=points 
		if win:
			self.winlist[state]=True

	def save(self):
		pickle.dump(self.points, open(self.filename, 'wb'))

	def try_load(self):
		if os.path.exists(self.filename):
			self.points=pickle.load(open(self.filename, 'rb'))
			return True
		else:
			return False
