import qnn
import importlib
import pdb

# 0,0 can go to 0,1, and 1,0
# 0,1 can lead to 1,1 in both cases, which is win
# 1,0 can lead to 0,1, but also to 2,2, which is lost. 

# ok i'll need a matrice anyway, but in some cases, i want some randomness, so a path is shorter but more risky (one chance on two)

#rahhhhh
global a

a=qnn.Qlearning()
a.define([1,1], 100)
a.define([2,2], -100)

def test(tries=10):

	for i in range(tries):
		state=[0,0]
		while state!=[1,1] and state!=[2,2]:
			points=None # if not none then will be sent to the remember function
			r=a.query(state, [0,1]) # 2 options possible
			oldstate=state
			if state==[0,0] and r==0:
				state=[0,1]
			if state==[0,0] and r==1:
				state=[1,0] 
			if state==[1,0] and r==0:
				state=[0,1] 
			if state==[1,0] and r==1:
				state=[2,2]
				points=-100
			if state==[0,1]:
				state=[1,1]
				points=100 
			a.remember(oldstate, state, points) 

		print("ended")



