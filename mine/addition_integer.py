#!/usr/bin/python3

import nn
import numpy as np
import os
import pdb
import random

count=2000
maxval=20
binsize=8



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

def datas(count, maxval, binsize):
	y = []
	for i in range(count):
		a = random.randrange(1,maxval)
		b = random.randrange(1,maxval)
		c = random.randrange(1,maxval)
		d = a + b +c  # the result expected is a+b because i know that works (later will try again somethg more complicated)
# le résultat attendu : d= a+b (d c'est le résultat)
		X1 = [a, b, c]
		if i == 0:
			X = np.array([X1])
		else:
			X = np.vstack([X, X1])
		y.append(d) 
	return X, y

# ok ?
def build_csv(count, maxval, binsize):
	X, y = datas(count, maxval, binsize)
	Xc = X/maxval 
	yc = convertytoS(y, binsize)  # as requested here
	np.savetxt('yd.csv', y, fmt='%s') 
	np.savetxt('Xd.csv', X, fmt='%s')
	np.savetxt('X.csv', Xc, fmt='%s',delimiter=',')
	np.savetxt('y.csv', yc, fmt='%s')

def example(count, maxval, binsize):
	build_csv()
	d, s = train()
	return d, s





def test(train, va, vb, vc):
	return nn.binary_to_int(train.FP(None, train.nn.syns, np.array([addition.convertXtoI([va,vb,vc], binsize)])) >= 0.5)

a=nn.Train()
if not os.path.exists('X.csv'):
	print("building datas")
	build_csv(count, maxval, binsize)
a.datas.raw.import_csv('./') 
a.datas.split(random=False)
a.nn.max_cpt=100000
a.nn.min_J=0
a.nn.min_J_cv=0.001
a.init_syns([16,16])
a.nn.filename='nnint.tmp'
a.nn.progress_display_size=10
a.try_to_load()



print("either do s = a.train()")
print("or test(a, 1, 2, 3)")
