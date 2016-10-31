#!/usr/bin/python3

# requierd :
#apt-get install python-numpy


# redo the digit number recognize in python, with the same number of layer.
# it is 400 neurons first layer, 25 neurons for the second layer, 10 for the third

# learn to read csv and convert into numpy stuff

import numpy as np
import math
from scipy.special import expit

class datas:
	def __init__(self, X, y):
		self.X = X
		self.y = y

def load(): 
#TODO : add ones in the first column
	X = np.genfromtxt('res/X.csv', delimiter=',')
	y = np.genfromtxt('res/y.csv', delimiter=',')
	y = int_to_array_bool(y)
#TODO here return 3 objects, training, cv and test. meaning take an optionnal parameter here to parse them (in percent, 3 values in an array i suppose
#so we just have then to train each of them, because train will use the vs thingy.
#lamba will be passed to train
#and there will be a function call train_with_cv, that will train and continue till cv is ok (i suppose). 
#well, we'll see that by hand for now.
#so we'll have train(data.train, minJ, maxcpt, lambda), and will return J (J will be calculated by another function, calcul, that take (data, synapses, var_shift)
#calcul return J and also the result, so we can use it with cross validation
	return datas(X,y)

class sdatas: # splitted datas
	def __init__(self):
		self.train=np.zeros(shape=(1,1))
		self.cv=np.zeros(shape=(1,1))
		self.test=np.zeros(shape=(1,1))

def split_in_third(X, y): # we should be able to give 3 values to change the splitting
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

def array_to_int(val): 
	l=np.array(range(val.shape[1])) 
	return [np.int(sum(a * l)) for a in val]

def add_ones(X): 
	return np.concatenate((np.array([1 for b in range(X.shape[0])])[None].T,X),1) 
	
def sigmoidGradient(X):
	g=expit(X) 
	return g*(1-g);

def train(X, y, neurons, min_J, max_cpt): # will train till J drops under a given value, or maxcount is reach. the array define the number of neurons. Now that is still to be done 
# return the weight for each neurons in an array (only one i think), plus J and count 
	np.random.seed
# we add ones here actually

	X = add_ones(X)
	m = X.shape[1]
	syn0 = 2*np.random.random((m,25)) - 1 
	syn1 = 2*np.random.random((26,10)) - 1
	l = 10
	error=999999
	for j in range(max_cpt):
		z1 = np.dot(X, syn0)
		a1 = add_ones(expit(z1))
		z2 = np.dot(a1, syn1)
		a2 = expit(z2)
		regul = l / (2*m) * (np.sum((syn1[:,2:]**2)) + np.sum((syn1[:,2:]**2))); # lambda has to be defined too... #TODO 
		#print(np.round(a2))

		#print(a2)
		J = np.sum((-y * np.log(a2) - (1 - y) * np.log(1 - a2))) / m; 
		J = J + regul;
	
		s2 = a2 - y # no idea why not the other way	

		error = np.sum(s2**2)
		print(error)
		if error < 0:
			exit

		print(array_to_int(y[0:70,]))
		print(array_to_int(a2[0:70,]>0.5))
		s1 = np.dot(s2,syn1.T) * (a1 * (1 - a1))
		s1 = s1[:,1:] # we don't want the error for the fixed 1, it's fixed

		d2 = a1.T.dot(s2)
		d1 = X.T.dot(s1) 

		#d2[1:,] = np.add(d2[1:,], l * syn1[1:,])
		#d1[1:,] = np.add(d1[1:,], l * syn0[1:,])

		syn1 -= d2 / m
		syn0 -= d1 / m

	return {'syn0':syn0, 'syn1':syn1} # should be a list here, that i will be able to used in the calcul function

class var_scale:
	def __init__(self):
		self.min=0
		self.max=0

def scale(data):
	vs=var_scale()
	vs.min=data.min()
	data=data-vs.min
	vs.max=data.max()
	data=data/vs.max
	data=data*2-1
	return [vs, data]

def unscale(data, vs):
	return (data + 1) / 2 * vs.max + vs.min

def calcul(X, r, vs=None): # syn will be a list of layers   # don't forget to use the 1/1+esomethg formula, and use the parameters of the scaling
# do that my way, with my !@#K functions

	syn0 = r['syn0']
	syn1 = r['syn1']
	X = add_ones(X)
	z1 = np.dot(X, syn0)
	a1 = add_ones(expit(z1))
	z2 = np.dot(a1, syn1)
	a2 = expit(z2) 

	if vs != None:
		a2 = unscale(a2,vs) 

	return a2
	


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
