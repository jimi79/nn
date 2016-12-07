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

class nn():
	def __init__(self)
		self.cpt=0
		self.min_cpt=1000
		self.min_data=1000
		self.max_data=5000
		self.X=[]
		self.y=[]
		self.nn=Train()
		self.nn.check_every_n_steps=1000 
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
		if len(self.X) > max_data:
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
			if not self.name is None:
				self.append_log("%s training" % self.name)
			self.nn.train()
			self.cpt=0 

class Qlearning():
	def __init__(self, max_state, max_action):
# max_action is all state included. it's just to prepare a input for the NN large enough
		self.array_wins=[]
		self.alpha=0.8 # factor with which i should use the max
		self.filename=None
		self.logfilename=None
		self.verbose=False 
		self.nna=Datas()
		self.nno=Datas()
		self.nna.min_J_cv=0.01 # easy to see what the outcome is from your own actions
		self.nno.min_J_cv=1.3 # harder to foresee what the opponent will play
		self.winlist=[] # index = True if win
		self.points=[] # index = points

	def set_name(self, name):
		self.name=name
		self.nn.name=name

	def append_log(self, text):
		if not self.logfilename is None:
			with open(self.logfilename, "a") as f:
				f.write("%s\n" % text)
		if self.verbose:
			print(text)

	def restart(self): 
		#self.history_status_ours=[] # will be feed by 'learn'
		#self.history_status_them=[] # will be feed by 'learn'
		#self.history_actions_ours=[] # will be feed by 'pick_action'
		pass

	def pick_action(self, state, list_actions):
		outputs={} # possible outputs for each action
		points={} # points available for a given action 
		s=""


#loop

# first nna, then check if win,feed the win list
# else: nno, and feed the list of points. update the status points too

# if win, put in wins list
# if points=None, put in donknow list
# if points>0, put in willwin
# if points<0, put in willlose

#then, pick action in win list if any
#else, pick action in willwin list (points>0) if any
#else, pick action in dontknow list (points=None) if any
#else, pick action if willose list (points<0) if any
#else : not possible, bug, raise exception

		n_winlist=[] # that action leads to a direct win # n_ like now
		n_willwin=[] # that action leads to a probable win
		n_maxpoint_pos=None # max points for a good action
		n_maxpoint_neg=None # max points for a bad action
		n_dontknow=[] # 
		n_willlose=[] # that action leads to a possible loss
		

		for i in list_actions: 
			a=np.zeros(self.max_action)
			a[i]=1 
			input_=np.concatenate([state, a]) # input should be a line
			res=self.nna.FP(input_)
			output=res>=0.5
			outputs[i]=output # it is a matrice here 
			new_state=array_to_integer(output)

			if not self.winlist.get(new_state)
				n_winlist.append(i)
			else:
				p=self.points.get(new_state)
				if not p is None:
					if p > 0:
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
			self.append_log("%d+%d=%d(%s points)," % (array_to_integer(state),i,new_state,p)) # %s to handle None


		action=None
		if len(n_winlist)!=0:
			action=random.choice(n_winlist)
			text="winning action %d" % action)
		else:
			if len(n_willwin)!=0:
				action=random.choice(n_willwin)
				text="i pick the best action %d for %d points" % (action, n_maxpoint_pos))
			else:
				if len(n_dontknow)!=0:
					action=random.choice(n_willwin)
					text="i pick the unknown action %d" % (action, n_maxpoint_neg)) 
				else:
					if len(n_willlose)!=0:
						action=random.choice(n_willlose)
						text="i pick the less bad action %d, for %d points" % (action, n_maxpoint_neg)) 
		self.append_log(text)
		return action

	def learn_action(self, oldstate, action, newstate): 
		a=np.zeros(self.max_action) 
		a[self.history_actions[-1]]=1
		input_=np.concatenate(oldstate, a]) # input should be a line
		output=newstate 
		nna.dd(input_, output) 
		nna.train_if_needed()
	
	def learn_opponent(self, oldstate, newstate, points=None): 
		input_=oldstate
		output=newstate 
		nno.dd(input_, output) 
		nno.train_if_needed()

	def learn_points(self, state, points, win): # which state is it ? Whatever, it's the state that you should aim for. But then, that is an issue, because that state can be reach either by me directly or by me and after the opponent plays
		state=array_to_integer(state)
		self.append_log("state %d is worth %d" % (array_to_integer(newstate), points))
		if self.verbose:
			print("I've got to remember that the state %d is worth %d" % (array_to_integer(newstate), points))
		self.array_points[state]=points 
		self.array_wins.append(state)

	def save(self):
		pickle.dump(self.array_points, open(self.filename, 'wb'))

	def try_load(self):
		if os.path.exists(self.filename):
			self.array_points=pickle.load(open(self.filename, 'rb'))
			return True
		else:
			return False
