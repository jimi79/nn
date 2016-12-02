import numpy as np

def temp_format(a): # convert an array with 00001000 into integer, but will works just for the 1-100 game
	b=np.where(a==1)[0]
	if len(b)>0:
		r=b[0]
	else:
		r=-1
	return r

def temp_format2(a):
	print("val %d, action %d, resul %d" % (temp_format(a[0:101]), temp_format(a[101:111])+1, temp_format(newstate)))
	
