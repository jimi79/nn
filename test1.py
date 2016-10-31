import ex3_in_python as e
from importlib import *
d=e.datas2() 
d.load('res1')
d.split(4500,250)
d.split()
s=d.train(0,5000,10)
r=d.check(d.cvset, s)
print("ratio on cv = %f" % r.ratio)
print("cost function on cv = %f" % r.J)

#d.print(d.raw.X[1000])

# create a new dataset, with a nn that can detect odd/even numbers
