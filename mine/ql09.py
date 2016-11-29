import numpy as np
import ql
import random

def play(val):
	act=100
	while act>val+11:
		act=act-11
	if act==val+11:
		act=act-1
	return act

def play_with_comp(val)
# bob is the perfect player
# alice is the Ql thingy
	a=ql.Qlearning()
	a.init(100,10)
	if random.random() >= 0.5:
		val=9 # bob starts
	else
		val=0 # alice starts
# val is the current value
	
	array_val=np.zeros(100)
	while val < 100:
		array_val[val]=1 
		action=a.pick_action(array_val, 10)

