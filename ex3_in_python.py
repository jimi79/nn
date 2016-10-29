#!/usr/bin/python3

# requierd :
#apt-get install python-numpy


# redo the digit number recognize in python, with the same number of layer.
# it is 400 neurons first layer, 25 neurons for the second layer, 10 for the third

# learn to read csv and convert into numpy stuff

import numpy as np

class datas:
	def __init__(self, X, y):
		self.X = X
		self.y = y

def load(): 
	X = np.genfromtxt('res/X.csv', delimiter=',')
	y = np.genfromtxt('res/y.csv', delimiter=',')
	y = int_to_array_bool(y)
	return datas(X,y)

class sdatas: # splitted datas
	def __init__(self):
		self.train=np.zeros(shape=(1,1))
		self.cv=np.zeros(shape=(1,1))
		self.test=np.zeros(shape=(1,1))

def split_in_third(X, y):
# not easy to shuffle that ...
	#c=np.c_[d.X.reshape(len(d.X), -1), d.y.reshape(len(d.y), -1)] 
	y_size = y.shape[1]
	c=np.c_[X, y]
	np.random.shuffle(c)
	cpt=round(c.shape[0] / 3)
	r=sdatas() 
	print(y_size)
	r.train=datas(c[0:cpt,0:-y_size], c[0:cpt,-y_size:]) 
	r.cv=datas(c[cpt:cpt*2,0:-y_size], c[cpt:cpt*2,-y_size:])
	r.test=datas(c[cpt*2:,0:-y_size], c[cpt*2:,-y_size:]) 
	return r 

def int_to_array_bool(y):
	a=range(int(min(y)), int(max(y))+1)
	return np.array([[i==j for j in a] for i in y]) # here i would like a numpy object

def train(X, y, neurons, min_J, max_cpt): # will train till J drops under a given value, or maxcount is reach. the array define the number of neurons. Now that is still to be done
# return the weight for each neurons in an array (only one i think), plus J and count 
	np.random.seed
	syn0 = 2*np.random.random((X.shape[1],25)) - 1 # am i stupid or what ? there are only 2 layers in the exercise, one to go from 400 to 25, another to go from 25 to 10
	syn1 = 2*np.random.random((25,10)) - 1
	for j in range(max_cpt):
		l1 = 1/(1+np.exp(-(np.dot(X,syn0)))) 
		l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
		print(l2)
		l2_delta = (y - l2)*(l2*(1-l2))
		l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
		syn1 += l1.T.dot(l2_delta)
		syn0 += X.T.dot(l1_delta) 
		tp=np.dot(np.dot(X[0].reshape(1,400),syn0),syn1)
		#print(l2_delta)
		#print(tp) 
	print("--------------------------")
	print(y)
	print(l2)
	return {'syn0':syn0, 'syn1':syn1} # should be a list here, that i will be able to used in the calcul function


def calcul(X, syns): # syn will be a list of layers
	pass 


#######
a="""def train_all_class():

# parameters should be training set, and checking set 
# i would like to give here a list of neurons, and loop till J is pretty low
# i would alos check if the number or percentage of results are close to 100%. But how does it relate to the cost function ? i should display it alongside with the percentage of right results
# actually, train should be 0 erreors, and validation should be pretty low too. i guess all should be 30% (train, validate, test) 

	syn0 = 2*np.random.random((400,25)) - 1
	syn1 = 2*np.random.random((25,10)) - 1

# i don't remember classification

	for j in range(60000):
		l1 = 1/(1+np.exp(-(np.dot(X,syn0))))
		l2 = 1/(1+np.exp(-(np.dot(l1,syn1))))
		l2_delta = (y - l2)*(l2*(1-l2))
		l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
		syn1 += l1.T.dot(l2_delta)
		syn0 += X.T.dot(l1_delta) """
