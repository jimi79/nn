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
	count = 10000
	maxv = 100
	for i in range(count):
		a = random.randrange(1,maxv)
		b = random.randrange(1,maxv)
		c = random.randrange(1,maxv)
		d = a * b + c
		X1 = [a, b, c]
		if i == 0:
			X = np.array([X1])
		else:
			X = np.vstack([X, X1])
		y.append(d) 
	return X, y


def build_csv():
	X, y = datas()
	Xc = convertXtoS(X, 16) 
	yc = convertytoS(y, 32) 
	np.savetxt('Xd.csv', X, fmt='%s')
	np.savetxt('yd.csv', y, fmt='%s') 
	np.savetxt('X.csv', Xc, fmt='%s')
	np.savetxt('y.csv', yc, fmt='%s')

def train(): 
	d=n.datas2() 
	d.load('.')
	d.split()
	s=d.train([32, 32], 0.01, 100000, 1)
	return d, s

def example():
	build_csv()
	d, s = train()
	return d, s

def test(d, s, a, b, c):
	return n.binary_to_int( d.FP(None, s, np.array([convertXtoI([a,b,c], 16)])) >= 0.5)


# syntax
# d, s = example()
# test(d, s, 12, 13, 14) # will tell what it thinks about that
