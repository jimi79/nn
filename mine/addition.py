#!/usr/bin/python3


import numpy as np
import random
import pdb
import nn_1 as n


def convertXtoS(inp, ds_inp):
	f = "{0:0" + str(ds_inp) + "b}"
	X = np.array([','.join(list(''.join([f.format(a) for a in b]))) for b in inp])
	return X

def convertXtoI(inp, ds_inp): # only one list of integer into an array of values
	f = "{0:0" + str(ds_inp) + "b}"
	s=[list(f.format(a)) for a in inp] 
	c = []
	for a in s:
		c = c + a 
	return [int(a) for a in c]


def convertytoS(out, ds_out):  
	f = "{0:0" + str(ds_out) + "b}"
	y = np.array([','.join(list(f.format(a))) for a in out])
	return y

def datas():
	y = []
	count = 5000 # number of examples i generate
	maxv = 100 # max value for a, b and c (let's start slow)
	for i in range(count):
		a = random.randrange(1,maxv)
		b = random.randrange(1,maxv)
		c = random.randrange(1,maxv)
		d = a + b +c  # the result expected is a+b because i know that works (later will try again somethg more complicated)
		X1 = [a, b, c]
		if i == 0:
			X = np.array([X1])
		else:
			X = np.vstack([X, X1])
		y.append(d) 
	return X, y

# ok ?
def build_csv():
	X, y = datas()
	Xc = convertXtoS(X, 8) 
	yc = convertytoS(y, 16)  # as requested here
	np.savetxt('Xd.csv', X, fmt='%s')
	np.savetxt('yd.csv', y, fmt='%s') 
	np.savetxt('X.csv', Xc, fmt='%s')
	np.savetxt('y.csv', yc, fmt='%s')

def train(): 
	d=n.datas2() 
	d.load('.')
	d.split() # third by default : 1/3 training, 1/3 cv, 1/3 test
	s=d.train([96,48], 0.001, 0.01, 10000000, 3) # the layout is here, 2 hidden layers of 64 neuros. last layer is to go from 64 to the expected 9 bits
	return d, s

def example():
	build_csv()
	d, s = train()
	return d, s

def test(d, s, a, b, c):
	return n.binary_to_int( d.FP(None, s, np.array([convertXtoI([a,b,c], 8)])) >= 0.5)


# syntax
# d, s = example()
# test(d, s, 12, 13, 14) # will tell what it thinks about that
