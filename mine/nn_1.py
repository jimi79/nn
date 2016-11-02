#!/usr/bin/python3

# requierd :
#apt-get install python3-numpy
#apt-get install python3-scipy


# redo the digit number recognize in python, with the same number of layer.
# it is 400 neurons first layer, 25 neurons for the second layer, 10 for the third

# learn to read csv and convert into numpy stuff

import numpy as np
import math
from scipy.special import expit
import sys

# we'll have an object, that will be called nn_params
# inside, we got raw, train, cv, shift, lambda, min_J, max_cpt, synapses. But we don't send that object directly to the train function
# raw is a datas class 
# train function return synapses


# class stand_datas
# raw, train, cv, stand_params
# class data
# X, y

def int_to_array_bool(y):
	a=range(int(min(y)), int(max(y))+1)
	return np.array([[i==j for j in a] for i in y]) # here i would like a numpy object

def array_to_int(val): 
	l=np.array(range(val.shape[1])) 
	return np.array([np.int(sum(a * l)) for a in val])

def int_to_binary(y, size):
	f = "{0:0" + str(size) + "b}"
	return np.array([list(f.format(x)) for x in y])

def binary_to_int(y): 
	return [int(''.join([str(int(bool(b))) for b in a]),2) for a in y]

def add_ones(X): 
	return np.concatenate((np.array([1 for b in range(X.shape[0])])[None].T,X),1) 
	
def sigmoidGradient(X):
	g=expit(X) 
	return g*(1-g); 

class network:
	def __init__(self):
		self.val = [] # an array of matrixes


class datas:
	def __init__(self, X = np.zeros((1,1)), y = np.zeros((1,1))):
		self.X = X
		self.y = y

class scale:
	def __init__(self):
		self.min=0
		self.max=0 

class datas2: # splitted datas
	def __init__(self):
		self.raw = datas()
		self.trainset = datas()
		self.cvset = datas()
		self.testset = datas()
		self.scale = None

	def load(self, directory): 
		self.raw.X = np.genfromtxt(directory + '/X.csv', delimiter=',')
		self.raw.y = np.genfromtxt(directory + '/y.csv', delimiter=',')
		#self.raw.y = int_to_array_bool(self.raw.y)

	def defscale(self):
		self.scale = scale()
		self.scale.min = self.raw.X.min()
		self.scale.max = self.raw.X.max()

	def rescale(self, data): 
		data = data - self.scale.min
		data = data / self.scale.max
		data = data * 2 - 1 
		return data

	def unscale(self, datas):
		return (datas + 1) / 2 * self.scale.max + self.scale.min

	def split(self, train_part=None, cv_part=None):
		y_size = self.raw.y.shape[1]
		c=np.c_[self.raw.X, self.raw.y]
		np.random.shuffle(c)
		cpt=round(c.shape[0] / 3)
		if train_part == None:
			train_part = cpt
		if cv_part == None:
			cv_part = cpt
		test_part = c.shape[0] - train_part - cv_part 

		self.trainset = datas(c[0:train_part,0:-y_size], c[0:train_part,-y_size:]) 
		self.cvset = datas(c[train_part:train_part + cv_part,0:-y_size], c[train_part:train_part + cv_part,-y_size:])
		self.testset = datas(c[train_part + cv_part:,0:-y_size], c[train_part + cv_part:,-y_size:]) 

	class layer:
		def __init__(self):
			z = None
			a = None
			s = None
			d = None

	class syn:
		def __init__(self):
			self.val = None
			self.val = None


	def FP(self, datalayers = None, syns = None, X=None): # X optionnal, in case we just want to run once 
		countlayers = len(syns)
		if not (X is None): 
			datalayers = [self.layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0
			datalayers[0].a = add_ones(X) # first activation is X. 

		for i in range(len(syns)):
			if i > 0: # we don't add ones each time for X
				datalayers[i].a = add_ones(datalayers[i].a)
			datalayers[i+1].z = np.dot(datalayers[i].a, syns[i].val)
			datalayers[i+1].a = expit(datalayers[i+1].z) 
		return(datalayers[-1].a)

	def BP(self, y, datalayers, syns, m, l):
		datalayers[-1].s = datalayers[-1].a - y
		for i in range(len(datalayers) - 2, -1, -1): # that will do datalayers - 2 up to 0
			s = datalayers[i+1].s
			if i < len(datalayers) - 2:
				s = s[:,1:]
			s = s.dot(syns[i].val.T)
			s = s * (datalayers[i].a * (1 - datalayers[i].a))
			datalayers[i].s = s 

		for i in range(len(datalayers) - 2, -1, -1):
			#a = add_ones(a)
			syns[i].d = datalayers[i].a.T.dot(datalayers[i+1].s) 
			if i < len(datalayers) - 2:
				syns[i].d = syns[i].d[:,1:] 
			syns[i].d += l * syns[i].val / m; # every synape is updated here
			syns[i].val -= syns[i].d / m

	def train(self, desc, min_J, min_J_on_cv, max_cpt, l, syns = None): # desc is only for the hidden layers 
		display_size = 30
		np.random.seed()
		y = self.trainset.y
		desc.append(y.shape[1])
		countlayers = len(desc) 
		datalayers = [self.layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0
		datalayers[0].a = self.trainset.X # first activation is X. 
		m = datalayers[0].a.shape[0] 
		fl = datalayers[0].a.shape[1]
		if syns is None:
			syns = [self.syn() for i in range(countlayers)] 
		for i in range(len(desc)): 
			syns[i].val = 2*np.random.random((fl + 1,desc[i])) - 1
			fl = desc[i]
		error=999999
		datalayers[0].a = add_ones(datalayers[0].a) # first activation is X. 
		oldacts = np.zeros(display_size)
		y2 = binary_to_int(y)
		y2s = binary_to_int(y[0:display_size])
		for cpt in range(max_cpt): 
			self.FP(datalayers, syns) # will update a 
			np.seterr(divide='ignore')
			J = np.sum((-y * np.log(datalayers[countlayers].a) - (1 - y) * np.log(1 - datalayers[countlayers].a))) / m; 
			np.seterr(divide='warn')
			if J <= min_J:
				break 
			self.BP(y, datalayers, syns, m, l) 
			#if (not np.array_equal(acts, oldacts)):
			#	acts = binary_to_int((datalayers[-1].a >= 0.5)[0:display_size]) 
			#	print("-------")
			#	print(' '.join(["{0:06d}".format(i) for i in y2s]))
			#	print(' '.join(["{0:06d}".format(i) for i in acts]))
			#	oldacts = acts
# too much lag at distance, i'll just display the ratio of what is wrong vs what is ok

			if (cpt % 100 == 0): 

				acts = binary_to_int((datalayers[-1].a >= 0.5)[0:display_size]) 

				if (not np.array_equal(acts, oldacts)):
					print("-------")
					print(' '.join(["{0:06d}".format(i) for i in y2s]))
					print(' '.join(["{0:06d}".format(i) for i in acts]))
					oldacts = acts 

				act = datalayers[-1].a >= 0.5
				act = binary_to_int(act) 
				oks = sum([ act==y2 for (act,y2) in zip(act, y2)] ) # i don't have nparrays at that point #### here is the ratio of ok results. i display that every 1000 training
				ratio = (oks / m)

				J_cv, oks_cv, ratio_cv = self.check(self.cvset, syns)
				print("After %i iterations, on training set, J = %d, ratio = %d" % (J, ratio, cpt))
				print("On cross-validation set, J = %d, ratio = %d" % (J_cv, ratio_cv))


		return syns

	def ascii(self, val): # val = self.trainset[0] for example
		a = val.reshape(20,20) > 0.5
		for i in a:
			for j in i:
				if j == True:
					sys.stdout.write("#")
				else:
					sys.stdout.write(".")
			print("") 

	def check(self, datas, syns): # datas is the type. Will return the cost function, and the number of different values
														# r should be in the object maybe. or not 
		countlayers = len(syns) 
		datalayers = [self.layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0
		datalayers[0].a = add_ones(datas.X) # first activation is X. 
		act = self.FP(datalayers, syns) # will update a 
		y = datas.y
		m = datas.X.shape[0]
		J = np.sum((-y * np.log(act) - (1 - y) * np.log(1 - act))) / m; 
#TODO : J seems to be wrong

		act = act >= 0.5
		act = array_to_int(act)
		y = array_to_int(y)
		oks = np.sum(act==y)
		ratio = np.sum(oks) / m
		return J, oks, ratio

			
		

# make an object of synapses, an array for the data, and a function load and save, so we can save a given state. because it takes some time to run
