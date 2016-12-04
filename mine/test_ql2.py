import importlib
import nn
import numpy as np
import pdb
import ql
import random 
import utils_ql


# the ql that is used. i make it global so i can study it easily

def init_ai(name): 
	ai=ql.Qlearning(101,10)
	ai.nn.init_syns([111,111],111,101) # 2 hidden layers 
	ai.min_data_to_train=1000 # default value
	ai.max_data_to_train=5000 # default value
	ai.min_cpt_since_last_train=1000 # don't train every time
	ai.cpt_since_last_train=0
	ai.nn.min_J_cv=0.05
	ai.verbose=False
	ai.nn.nn.verbose=False
	ai.nn.nn.filename="%s.tmp" % name
	ai.nn.load_synapses() 
	ai.nn.nn.max_cpt=100
	ai.name=name
	return ai

alice=init_ai("alice")
bob=init_ai("bob") 

def play(val):
	act=100
	while act>val+11:
		act=act-11
	if act==val+11:
		act=act-1
	return act-val 

def play_AI(AI, AI2, actions, vals, verbose=True):
# if vals got enough values, we can have it learning
# each time alice plays, bob learns.
# exception : if one wins, both learn. Will be handled in main loop. maybe. i don't know. it's complicayed
	val=vals[-1] 
	array_val=np.zeros(101)
	array_val[val]=1 
	action=AI.pick_action(array_val, 10) # action 0 alice plays 1, action 1 alice plays 2, etc 
	if verbose:
		print("%s plays %d" % (AI.name, action+1))
	actions.append(action)
	newval=val+action+1
	if newval>100:
		newval=100
	vals.append(newval)

	if len(vals)>3: # now we can learn the other one what happend
		oldval=np.zeros(101)
		oldval[vals[-3]]=1
		oldvalb=np.zeros(101)
		oldvalb[vals[-1]]=1
		oldaction=actions[-3]
		if verbose:
			print("I tell %s what happened. val was %d before they played %d, and now it's %d" % (AI2.name, utils_ql.temp_format(oldval), oldaction, utils_ql.temp_format(oldvalb)))
		points=None
		if newval==100:
			points=-100
		AI2.learn(oldval, 10, oldaction, oldvalb, points) 

	# exception : if AI wins, then i should notify it too, because nobody will otherwise
	if newval==100:
		oldval=np.zeros(101)
		oldval[vals[-2]]=1
		oldvalb=np.zeros(101)
		oldvalb[vals[-1]]=1
		oldaction=actions[-1]
		if verbose:
			print("I tell %s what happened. val was %d before they played %d, and now it's %d" % (AI.name, utils_ql.temp_format(oldval), oldaction, utils_ql.temp_format(oldvalb)))
		AI.learn(oldval, 10, oldaction, oldvalb, 100) 

	return actions,vals



def play(verbose):
# bob is the perfect player
# alice is the Ql thingy
	bob.verbose=verbose
	alice.verbose=verbose
	bob.nn.nn.verbose=verbose
	alice.nn.nn.verbose=verbose

	if random.random() >= 0.5: # bob starts less often, giving more chances for alice to win
		alicestart=True
	else:
		alicestart=False

	vals=[0] 
	actions=[] # list of actions

	alicewin=False
	bobwin=False
	winner=""
	while winner=="":
		if alicestart:
			actions,vals=play_AI(alice, bob, actions, vals, verbose=verbose) 
		if vals[-1]==100:
			winner="alice" 
		if winner=="":
			actions,vals=play_AI(bob, alice, actions, vals, verbose=verbose)
			if vals[-1]==100:
				winner="bob" 
		alicestart=True # next turn we play normally 
	if verbose:
		print ("%s win" % winner)


def loop(cpt):
	for i in range(cpt):
		play(cpt<200)
		print(i)
	
