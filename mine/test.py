import importlib
import nn
import addition
import numpy as np


def test(train, a, b, c):
	return nn.binary_to_int(train.FP(None, train.nn.syns, np.array([addition.convertXtoI([a,b,c], 8)])) >= 0.5)

#addition.build_csv()

a=nn.Train()
a.datas.raw.import_csv('./') 
a.datas.split(random=False)
a.nn.max_cpt=100000
a.nn.min_J=0.001
a.nn.min_J_cv=0.001
a.init_syns([32,32])
a.nn.progress_display_size=10
s = a.train()

print(test(a, 1,2,3))

