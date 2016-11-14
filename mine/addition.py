#!/usr/bin/python3

import numpy as np
import random
import pdb


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
	count = 100000 # number of examples i generate # <- 1000 exemples. cad le NN va prendre 1000 exemples, utilisé les poids de ses neurones (le premier tour les poids sont juste random). pour chaque exemple il sort le résultat, et pour les 1000 on regarde la diff, et on applique la diff sur les poids, de tel sorte que à chq tour il prévoit kkchose de plus proche de ce qui est attendu
# je te déatil après si tu veux

	maxv = 50 # max value for a, b and c (let's start slow) # valeur max pour a, b et c
	for i in range(count):
		a = random.randrange(1,maxv)
		b = random.randrange(1,maxv)
		c = random.randrange(1,maxv)
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
def build_csv():
	X, y = datas()
	Xc = convertXtoS(X, 8) 
	yc = convertytoS(y, 8)  # as requested here
	np.savetxt('Xd.csv', X, fmt='%s')
	np.savetxt('yd.csv', y, fmt='%s') 
	np.savetxt('X.csv', Xc, fmt='%s')
	np.savetxt('y.csv', yc, fmt='%s')

def example():
	build_csv()
	d, s = train()
	return d, s
