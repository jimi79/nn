#!/usr/bin/python3

# requierd :
#apt-get install python-numpy


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
		self.raw.y = int_to_array_bool(self.raw.y)

	def rescale(self): 
		self.scale = scale()
		self.scale.min = self.raw.X.min()
		self.raw.X = self.raw.X - self.scale.min
		self.scale.max = self.raw.X.max()
		self.raw.X = self.raw.X / self.scale.max
		self.raw.X = self.raw.X * 2 - 1 

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

	def train(self, min_J, max_cpt, l): # and after that, it will take an array as a neural network, once i've played enough with it
		np.random.seed 
		X = add_ones(self.trainset.X)
		y = self.trainset.y
		m = X.shape[0]
		syn0 = 2*np.random.random((X.shape[1],25)) - 1 
		syn1 = 2*np.random.random((26,10)) - 1
		error=999999
		for cpt in range(max_cpt): 
			z1 = np.dot(X, syn0)
			a1 = add_ones(expit(z1))
			z2 = np.dot(a1, syn1)
			a2 = expit(z2)
# annoying, i had a2 = 0 (or 1), and thus log failed
			np.seterr(divide='ignore')
			J = np.sum((-y * np.log(a2) - (1 - y) * np.log(1 - a2))) / m; 
			np.seterr(divide='warn')
			if J < min_J:
				break
			s2 = a2 - y 
			s1 = np.dot(s2,syn1.T) * (a1 * (1 - a1))
			s1 = s1[:,1:] 
			d2 = a1.T.dot(s2)
			d1 = X.T.dot(s1) 
			d2[1:,] += l * syn1[1:,] / m;
			d1[1:,] += l * syn0[1:,] / m; 
			syn1 -= d2 / m
			syn0 -= d1 / m 

			if (cpt % 100 == 0):
				a2 = a2 >= 0.5
				act = array_to_int(a2)
				y2 = array_to_int(y)
				res = act == y2
				errs = np.nonzero(1-res)[0]
				ratio = 1 - (errs.shape[0] / y2.shape[0])
				print("J = %f" % J)
				print("ratio = %f" % ratio)
				print("cpt = %i" % cpt)
				print(y2[0:30])
				print(act[0:30])
				#if errs.shape[0] > 0:
				#	err = errs[0]
				#	self.print(self.trainset.X[err])
				#	print("Expected : %i" % (y2[err] + 1)) # because our array goes from 0 to 9 
				if ratio == 1:
					break 
				# i would like to locate the first wrong one, and display it in ascii art, and then print what the computer thought it was 
		return {'syn0':syn0, 'syn1':syn1} # should be a list here, that i will be able to used in the calcul function

	def calcul(self, X, s):
		syn0 = s['syn0']
		syn1 = s['syn1']
		X = add_ones(X)
		z1 = np.dot(X, syn0)
		a1 = add_ones(expit(z1))
		z2 = np.dot(a1, syn1)
		a2 = expit(z2) 
		if self.scale != None:
			a2 = self.unscale(a2) 
		return a2

	def print(self, val): # val = self.trainset[0] for example
		a = val.reshape(20,20) > 0.5
		for i in a:
			for j in i:
				if j == True:
					sys.stdout.write("#")
				else:
					sys.stdout.write(".")
			print("") 

	class results:
		def __init__(self):
			self.J = 0
			self.oks = 0
			self.ratio = 0

	def check(self, datas, s): # datas is the type. Will return the cost function, and the number of different values
														# r should be in the object maybe. or not
		m = datas.X.shape[0]
		act = self.calcul(datas.X, s)
		act = np.array(act)
		y = datas.y
		J = np.sum((-y * np.log(act) - (1 - y) * np.log(1 - act))) / m; 

		r = self.results()
		r.J = J
		act = act >= 0.5
		act = array_to_int(act)
		y = array_to_int(y)
		r.oks = np.sum(act==y)
		r.ratio = np.sum(r.oks) / m
		return r

			
		

