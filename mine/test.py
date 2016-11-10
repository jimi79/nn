import importlib
import nn
import addition
import numpy as np



def test(train, a, b, c):
	return nn.binary_to_int(train.FP(None, train.nn.syns.syns, np.array([addition.convertXtoI([a,b,c], 8)])) >= 0.5)


addition.build_csv()

a=nn.Train()
a.datas.raw.import_csv('./')
a.datas.split()
a.nn.max_cpt=100000
a.nn.min_J=0.001
a.nn.min_J_cv=0.001
a.init_syns([16,16])
a.nn.progress_display_size=10
s = a.train()

# addition.test(d, s, 12, 13, 14) # will tell what it thinks about that

print(test(a, 1,2,3))


# all that code should be in addition.py. there is one specific unit for the work (addition), and one commmon unit (nn.py).
