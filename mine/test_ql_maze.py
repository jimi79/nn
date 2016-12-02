import random

def run(old, action): # will tell you the new status
	new=None
	if old==0 and action==0:
		new=1
	if old==0 and action==1:
		new=4
	if old==1 and action==0:
		new=2
	if old==1 and action==1:
		r=random.random()
		if r>0.9:
			new=7
		else:
			new=6
	if old==2 and action==0:
		new=5
	if old==2 and action==1:
		new=6
	if old==5:
		new=6
	if old==4:
		new=7

	return new
	
