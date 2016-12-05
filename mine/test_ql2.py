import importlib
import nn
import numpy as np
import pdb
import ql
import random 
import utils_ql



def init_ai(name): 
	ai=ql.Qlearning(102,10)
	ai.nn.init_syns([112,112],112,102) # 2 hidden layers 
	ai.min_data_to_train=5000 # default value
	ai.max_data_to_train=7000 # default value
	ai.min_cpt_since_last_train=1000 # don't train every time
	ai.nn.nn.save_every_n_steps=-1
	ai.cpt_since_last_train=0
	ai.nn.min_J_cv=0.05
	ai.verbose=True
	ai.nn.verbose=True
	ai.nn.nn.filename="%s.tmp" % name
	ai.nn.load_synapses() 
	ai.nn.nn.max_cpt=100
	ai.nn.nn.check_every_n_steps=10
	ai.name=name
	return ai

# the ql that is used. i make it global so i can study it easily
alice=init_ai("alice")
bob=init_ai("bob") 
all_actions=list(range(1,11))

def play_AI(AI, AI2, actions, vals, verbose=True):
# if vals got enough values, we can have it learning
# each time alice plays, bob learns.
# exception : if one wins, both learn. Will be handled in main loop. maybe. i don't know. it's complicayed
	val=vals[-1] 
	array_val=np.zeros(102)
	array_val[val]=1 

	max_action=min(100-val,10) 
	av_actions=list(range(max_action)) 

	if verbose:
		print("I ask %s to play" % AI.name)
	action=AI.pick_action(array_val, av_actions) 
	actions.append(action)
	newval=val+action+1
	if newval>100:
		newval=100
	vals.append(newval)
	if verbose:
		print("%s plays %d, value is now %d" % (AI.name, action+1, newval))

	if len(vals)>3: # now we can learn the other one what happend
		oldval=np.zeros(102)
		oldval[vals[-3]]=1
		oldvalb=np.zeros(102)
		oldvalb[vals[-1]]=1
		oldaction=actions[-3]
		points=None
		if newval==100:
			points=-100
			oldvalb=np.zeros(102)
			oldvalb[101]=1
		if verbose:
			print("I tell %s what happened. val was %d before they played %d, and now it's %d" % (AI2.name, utils_ql.temp_format(oldval), oldaction+1, utils_ql.temp_format(oldvalb)))
		AI2.learn(oldval, 10, oldaction-1, oldvalb, points) 

	# exception : if AI wins, then i should notify it too, because nobody will otherwise
	if newval==100:
		oldval=np.zeros(102)
		oldval[vals[-2]]=1
		oldvalb=np.zeros(102)
		oldvalb[vals[-1]]=1 # all wrong here, plz fix
		oldaction=actions[-1] # that is still an integer
		if verbose:
			print("I tell %s what happened. val was %d before they played %d, and now it's %d" % (AI.name, utils_ql.temp_format(oldval), oldaction+1, utils_ql.temp_format(oldvalb)))
		AI.learn(oldval, 10, oldaction-1, oldvalb, 100) 
	return actions,vals 


def play(verbose):
# bob is the perfect player
# alice is the Ql thingy
	bob.verbose=verbose
	alice.verbose=verbose
	bob.nn.nn.verbose=True
	alice.nn.nn.verbose=True

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
	verbose=cpt<=200
	for i in range(cpt):
		play(verbose)
		if not verbose:
			print(i)
	
