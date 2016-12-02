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

def play_with_comp(verbose, withAI, withCSV, X, y):
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
			a=np.zeros(10) 
			a[action]=1
			input_=np.concatenate([array_val, a]) # input should be a line
			if X==None:
				X=np.array([input_])
			else:
				X=np.append(X, np.array([input_]), axis=0)
			if y==None:
				y=np.array([array_val2])
			else:
				y=np.append(y, np.array([array_val2]), axis=0)

	if verbose:
		if alicewin:
			print("alice win")
		if bobwin:
			print("bob win")

	return X,y

def loop_with_comp(count, withAI=True, withCSV=False):
	X=None
	y=None
	if count==1:
		verbose=True
		myql.verbose=True
	else:
		verbose=False
		myql.verbose=False
	for i in range(count):
		X,y=play_with_comp(verbose, withAI, withCSV, X, y) 

	if withCSV: 
		np.savetxt('X.csv', X, fmt='%.0f', delimiter=',') 
		np.savetxt('y.csv', y, fmt='%.0f', delimiter=',') 

		text=["%d,%d,%d" % (utils_ql.temp_format(X[i,0:101]), utils_ql.temp_format(X[i,101:])+1, utils_ql.temp_format(y[i])) for i in range(y.shape[0])]
		np.savetxt('text', text, fmt='%s') 

def get_nn():
	nntmp=nn.Train()
	print("loading datas")
	nntmp.datas.raw.import_csv('./') 
	nntmp.datas.split(random=True,train_part=0.8)
	nntmp.nn.max_cpt=100
	nntmp.nn.min_J=0.001
	nntmp.nn.min_J_cv=0.001
	nntmp.nn.verbose=True
	nntmp.init_syns_for_trainset([101])
	nntmp.filename='nnqltmp.syn'
	nntmp.check_every_n_steps=1
	nntmp.save_every_n_steps=10
	return nntmp

def test(nn, val, action):
	a=np.zeros(101)
	a[val]=1
	b=np.zeros(10)
	b[action]=1
	input_=np.concatenate([a, b])
	r=nn.FP(input_)
	print(utils_ql.temp_format(r))

#nn=get_nn()
#nn.load_synapses()
#test(nn,1,2)
#test(nn,13,4)
#test(nn,85,2)
