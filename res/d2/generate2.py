#!/usr/bin/python3

import numpy as np
#a=np.array([[','.join('{0:024b}'.format(a)), a%2] for a in range(233)])
a=np.array([[','.join('{0:024b}'.format(a)), int(a>123 and a < 2245)] for a in range(10000)])

np.savetxt('X.csv', a[:,0], fmt='%s')
np.savetxt('y.csv', a[:,1], fmt='%s')
