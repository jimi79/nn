import ex3_in_python as e
from importlib import *
import numpy as np
d=e.datas2() 
d.load('res/d2')
d.defscale()

d.raw.X = d.rescale(d.raw.X)
d.split()
s=d.train(0,50000,10)
r=d.check(d.cvset, s)
print("ratio on cv = %f" % r.ratio)
print("cost function on cv = %f" % r.J)

np.set_printoptions(precision=3,linewidth=200, threshold=2000)

def test(val):
	a = [','.join('{0:024b}'.format(val)).split(',')]
	a = np.array([[int(j) for j in i] for i in a])
	a = d.rescale(a)
	r = d.calcul(a, s)
	return r

print(test(122))


