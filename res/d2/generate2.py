#!/usr/bin/python3

import numpy as np
a=np.array([[','.join('{0:024b}'.format(a)), a%2] for a in range(25340)])

np.savetxt('X.csv', a[:,0], fmt='%s')
np.savetxt('y.csv', a[:,1], fmt='%s')
