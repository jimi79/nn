import qnn
import importlib
import pdb

def init():
  Q=Qlearning()
	Q.define_points([1,1],100)
	Q.define_points([2,2],-100)
	return Qlearning()	


def play_once(Q): # Q is a Qlearning object
	position=[1,1] # those are coordinates
	lost=False
	win=False
	while (not lost) and (not win):
		if position=[0,0]:
			actions=[0,1] # possible actions are 0 and 1. This is a list of possible actions here 
		if position=[0,1]:
			actions=[0,1] 
		if position=[1,2]:
			actions=[0] 
		if position=[1,0]:
			actions=[0] 

		action=Q.request(position, actions)

		old_pos=position

		if position=[0,0]:
			if action=0:
				position=[0,1]
			if action=1:
				position=[1,0]
		if position=[0,1]:
			if action=0:
				position=[0,2]
			if action=1:
				position=[0,1]
		if position=[0,2]:
			position=[1,1]
		if position=[1,0]:
			a=random.randrange(4)
			if a<3:
				position=[1,1]
			else:
				position=[2,2]

		Q.learn(old_position,action,position)

		if position=[1,1]:
			win=True
		if position=[2,2]:
			lost=True


	if win:
		print("Win") 
	if lost:
		print("Lost") 
