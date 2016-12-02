import importlib
import nn
import numpy as np
import pdb
import ql
import random 
import utils_ql


# the ql that is used. i make it global so i can study it easily
myql=ql.Qlearning(101,10)
myql.nn.init_syns([111,111],111,101) # 2 hidden layers 


def play(val):
	act=100
	while act>val+11:
		act=act-11
	if act==val+11:
		act=act-1
	return act-val 

def play_with_comp(verbose, withAI, withCSV, X, y, winners):
# bob is the perfect player
# alice is the Ql thingy

	if random.random() >= 0.5:
		val=9 # bob starts
		if verbose:
			print("bob starts with %d" % val)
	else:
		val=0 # alice starts
		if verbose:
			print("alice starts")
# val is the current value
	
	array_val=np.zeros(101)
	alicewin=False
	bobwin=False
	winner=""
	while val < 100:
		array_val=np.zeros(101)
		array_val[val]=1  
		if withAI:
			action=myql.pick_action(array_val, 10) # action 0 alice plays 1, action 1 alice plays 2, etc
		else:
			action=random.randrange(10)
		alice_number=action+1

		val=val+alice_number
		if verbose:
			print("alice played %d, score is %d" % (alice_number,val))
		if val != 100: 
			bob_number=play(val)
			val=val+bob_number
			if verbose:
				print("bob played %d, score is %d" % (bob_number,val))
			if val==100:
				bobwin=True
		else:
			alicewin=True

		array_val2=np.zeros(101)
		array_val2[val]=1 
		if withAI:
			myql.learn(array_val, 10, action, array_val2)

		if withCSV:
			if alicewin:
				winner="alice"
			if bobwin:
				winner="bob"
			a=np.zeros(10) 
			a[action]=1
			input_=np.concatenate([array_val, a]) # input should be a line
			if X is None:
				X=np.array([input_])
			else:
				X=np.append(X, np.array([input_]), axis=0)
			if y is None:
				y=np.array([array_val2])
			else:
				y=np.append(y, np.array([array_val2]), axis=0)
			if winners is None:
				winners=np.array([winner])
			else:
				winners=np.append(winners, np.array([winner]))

	if verbose:
		if alicewin:
			print("alice win")
		if bobwin:
			print("bob win")

	return X,y,winners

def loop_with_comp(count, withAI=True, withCSV=False):
	X=None
	y=None
	winners=None
	if count==1:
		verbose=True
		myql.verbose=True
	else:
		verbose=False
		myql.verbose=False
	for i in range(count):
		X,y,winners=play_with_comp(verbose, withAI, withCSV, X, y, winners) 

	if withCSV: 
		np.savetxt('X.csv', X, fmt='%.0f', delimiter=',') 
		np.savetxt('y.csv', y, fmt='%.0f', delimiter=',') 

		text=["%d,%d,%d,%s" % (utils_ql.temp_format(X[i,0:101]), utils_ql.temp_format(X[i,101:])+1, utils_ql.temp_format(y[i]), winners[i]) for i in range(y.shape[0])]
		np.savetxt('text', text, fmt='%s') 

def get_nn():
	nntmp=nn.Train()
	print("loading datas")
	nntmp.datas.raw.import_csv('./') 
	nntmp.datas.split(random=True,train_part=int(0.8*nntmp.datas.raw.X.shape[0]))
	nntmp.nn.max_cpt=10000
	nntmp.nn.min_J=0.001
	nntmp.nn.min_J_cv=0.001
	nntmp.nn.verbose=True
	nntmp.init_syns_for_trainset([101])
	nntmp.nn.filename='nnqltmp.syn'
	nntmp.nn.check_every_n_steps=10
	nntmp.nn.lambda_=1
	nntmp.nn.save_every_n_steps=100
	nntmp.nn.lambda_=0 # maybe it will learn faster that way..
	return nntmp

def test(nn, val, action):
	a=np.zeros(101)
	a[val]=1
	b=np.zeros(10)
	b[action]=1
	input_=np.concatenate([a, b])
	r=nn.FP(input_)
	return r, utils_ql.temp_format(r>=0.5)

#nn=get_nn()
#nn.load_synapses()
#test(nn,1,2)
#test(nn,13,4)
#test(nn,85,2)


# idea : if that fucking thing can learn, eventually it will avoid loosing positions, and it will win.

# maybe i need to do FP/BP till it learns at least why i'm trying to have it learning, only to do it once for each value. Which is the same than looping anyway

# in short : i need the NN to learn faster, and then do the qlearning algo so that it will at least try to avoid situations during which it loses

# ok, if that thing learn, then i'll create a unit in between, that will :
# ask the thing to play
# add the result to an array
# BP the whole array
# if the array is bigger than 10000, then remove the 100 first item.
# because if i learn only the last one, then either alpha is high (1) and i will just make the NN dumber for all possibilities except the last one
# either alpha is too low and that won't be any help to learn anythg.
# i want the NN to be able to figure out a situation wihtout forgetting everythg else

