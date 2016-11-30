#!/usr/bin/python3

# requierd :
#apt-get install python3-numpy
#apt-get install python3-scipy


# redo the digit number recognize in python, with the same number of layer.
# it is 400 neurons first layer, 25 neurons for the second layer, 10 for the third

# learn to read csv and convert into numpy stuff

import json
import math
import numpy as np
import os
import pickle
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

class Set: 
	def __init__(self, X = np.zeros((1,1)), y = np.zeros((1,1))):
		self.X = X
		self.y = y

	def import_csv(self, directory): 
		self.X = np.genfromtxt(directory + '/X.csv', delimiter=',')
		self.y = np.genfromtxt(directory + '/y.csv', delimiter=',') 

class Layer:
	def __init__(self):
		z = None # output of the matrice multiplication
		a = None # after the sigmoid
		s = None # difference sigma
		d = None # delta to apply to synapses

class Datas: 
	def __init__(self):
		self.scale = None
		self.raw = Set()
		self.trainset = Set()
		self.cvset = Set()
		self.testing = Set() 

	def defscale(self, datas):
		self.scale = scale()
		self.scale.min = datas.min()
		self.scale.max = datas.max()

	def rescale(self, data): 
		data = data - self.scale.min
		data = data / self.scale.max
		data = data * 2 - 1 
		return data

	def unscale(self, datas):
		return (datas + 1) / 2 * self.scale.max + self.scale.min

	def split(self, train_part=None, cv_part=None, random=True):
		y_size = self.raw.y.shape[1]
		c=np.c_[self.raw.X, self.raw.y]
		if random:
			np.random.shuffle(c)
		cpt=round(c.shape[0] / 3)
		if train_part == None:
			train_part = cpt
		if cv_part == None:
			cv_part = cpt
		test_part = c.shape[0] - train_part - cv_part 

		self.trainset = Set(c[0:train_part,0:-y_size], c[0:train_part,-y_size:]) 
		self.cvset = Set(c[train_part:train_part + cv_part,0:-y_size], c[train_part:train_part + cv_part,-y_size:])
		self.testset = Set(c[train_part + cv_part:,0:-y_size], c[train_part + cv_part:,-y_size:]) 

class Syns: 

	class Syn:
		def __init__(self):
			self.vals = None

	def __init__(self, layers=[], in_size=1):
		np.random.seed()
		self.vals = [None for i in range(len(layers))]
		for i in range(len(layers)): 
			self.vals[i] = 2*np.random.random((in_size + 1,layers[i])) - 1
			in_size = layers[i] 

		countlayers = len(self.vals) # we count the number of synapses to define our laters. Layers arejust for FP and BP
		datalayers = [Layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0




	def save(self, filename):
		pickle.dump(self.vals, open(filename, "wb"))

	def load(self, filename):
		self.vals = pickle.load(open(filename, "rb"))

class NN:
	def __init__(self):
		self.check_every_n_steps = 100
		self.save_every_n_steps = 1000
		self.dataset = None 
		self.min_J = 0.001
		self.min_J_cv = 0.01 
		self.max_cpt = -1 # handle that case stupid
		self.lambda_=3 # lambda default value
		self.alpha=1 #alpha default value
		self.filename='nn.tmp.dat' # file to save progress (and everythg else)
		self.verbose=True
		self.syns=None

class Train: 
	def __init__(self): 
		self.nn = NN()
		self.datas = Datas() 

	def try_to_load(self):
		if os.path.exists(self.nn.filename):
			if self.nn.verbose:
				print("loading temp synapses values")
			self.nn.syns.load(self.nn.filename) 

	def init_syns(self, size):
		size.append(self.datas.trainset.y.shape[1]) 
		self.nn.syns = Syns(size, self.datas.trainset.X.shape[1])

	def FPdl(self, X): # FP that return the whole datalayer
		syns=self.nn.syns
		countlayers = len(syns.vals)
		datalayers = [Layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0
		datalayers[0].a = X 
		for i in range(len(syns.vals)):
			datalayers[i].a = add_ones(datalayers[i].a)
			datalayers[i+1].z = np.dot(datalayers[i].a, syns.vals[i])
			datalayers[i+1].a = expit(datalayers[i+1].z) 
		#return(datalayers[-1].a)
		return(datalayers) # will be used for the BP

	def FP(self, X): # result is only the last layer, not the intermediate results
		return self.FPdl(X)[-1].a

	def BP(self, y, datalayers):
		m=y.shape[0]
		datalayers[-1].s = datalayers[-1].a - y
		syns=self.nn.syns
		for i in range(len(datalayers) - 2, -1, -1): # that will do datalayers - 2 up to 0
			s = datalayers[i+1].s
			if i < len(datalayers) - 2:
				s = s[:,1:]
			s = s.dot(syns.vals[i].T)
			s = s * (datalayers[i].a * (1 - datalayers[i].a))
			datalayers[i].s = s 

		diffs = [np.zeros(syns.vals[a].shape) for a in range(len(syns.vals))]
		for i in range(len(datalayers) - 2, -1, -1):
			diffs[i] = datalayers[i].a.T.dot(datalayers[i+1].s) 
			if i < len(datalayers) - 2:
				diffs[i] = diffs[i][:,1:] 
			diffs[i] += self.nn.lambda_ * syns.vals[i] / m; # every synape is updated here
			syns.vals[i] -= self.nn.alpha * (diffs[i] / m)

	def cost_function(self, yguess, yexpected):
		m=yguess.shape[0] 
		#J = np.sum((-yexpected * np.log(yguess) - (1 - yexpected) * np.log(1 - yguess))) / m # i've got divided by 0 here in log
		J = np.sum((yexpected - yguess)**2) / m # i've got divided by 0 here in log 
		return J

	def train(self): # desc is only for the hidden layers 
		self.try_to_load()
		y = self.datas.trainset.y
		error=999999
		cpt = 0
		while ((cpt < self.nn.max_cpt) or (self.nn.max_cpt == -1)):
			cpt = cpt + 1
			datalayers=self.FPdl(self.datas.trainset.X) # will update a 
			np.seterr(divide='ignore')
			self.BP(y, datalayers)  # removed m, because that's dedudnant with a count of y or datalayers, remvoed l because in object self or params.
			if self.nn.check_every_n_steps!=None: 
				if (cpt % self.nn.check_every_n_steps == 0): 
					J = self.cost_function(datalayers[-1].a, y)
					#np.seterr(divide='warn')
					if J <= self.nn.min_J:
						break 
					J_cv = self.check(self.datas.cvset) 
					if self.nn.verbose:
						print("Jtrain = %f, Jcv = %f" % (J, J_cv))
			if (cpt % self.nn.save_every_n_steps == 0): 
				if self.nn.verbose:
					print("saved")
				self.nn.syns.save(self.nn.filename) 
		return self.nn.syns.vals

	def ascii(self, val): # val = self.trainset[0] for example
		a = val.reshape(20,20) > 0.5
		for i in a:
			for j in i:
				if j == True:
					sys.stdout.write("#")
				else:
					sys.stdout.write(".")
			print("") 

	def check(self, datas): # datas is the type. Will return the cost function, and the number of different values
														# r should be in the object maybe. or not 
		syns=self.nn.syns
		countlayers = len(syns.vals) 
		datalayers = [Layer() for i in range(countlayers + 1)] # layer0 = X, layer1 = layer0 * syn0
		yguess = self.FP(datas.X) # will update a 
		y = datas.y
		m = datas.X.shape[0]
		J = self.cost_function(y, yguess)
		return J


# to use it : 
# nn.init
# nn.syns=syns([16,16], 8)
# nn.train()

# and to reprise : nn.train() should be ok
# synapses are in TrainParams ... i have no f@#%@ idea where they should be
# TrainParams shoule be nn, and nn should be train, that train a nn versus datas

