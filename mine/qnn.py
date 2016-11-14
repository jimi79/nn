#!/usr/bin/python3

#ok, so, there should be two functions in my class. one that ask 'what do you want to do', second that says 'well, you did that and that happened'. one extra parameter to say 'and you win'. Because it should win at the end

import numpy as np 
import random

class Qlearning:

	class State:
		def __init__(self, ident, points=0):
			self.ident=ident # identification of that state
			self.leans_to=[] # array of states that this one can lead to
			self.points=points # value of that state 

	def __init__(self):
		self.states=[] # will be filled with states

	def __test(self): # that is private :)
		print("private") 

	def find(self, ident):
		for i in self.states:
			print("looking")
			print(i.ident)
			if i.ident == ident:
				print("found")
				return i # return a state
				break
		return None

	def define(self, state, points=None): # can be used to add the win value 
		i=self.find(state)
		if points!=None:
			if i!=None:
				i.points = points
			else: 
				self.states.append(self.State(state, points))
		print("defining")
		print(state)
		print(points)

	def query(self, state, possible_actions):
		self.define(state)
		return self.best_possible_action(possible_actions)
# return here the best possible_action

	def best_possible_action(self, possible_actions):
		return possible_actions[random.randrange(0,len(possible_actions))] # i should use a NN for that, based on the score


	def remember(self, old_state, new_state, action):
# first, if we win, then we write 100 in the new old_state new_state combination
# if we don't win, then we update the path we took, meaning the old_state / new_state cell with the gamma * max(new_state) row (meaning all possible new states + 1)
# we do it only here, because we know that going from old to new is possible
# it's here that we'll reinforce old_state thing
		self.define(new_state) # we don't know if that new state is good now
# but we know the old_state has to be updated
		i=self.find(old_state) # shoulod exists
		sumval=0
		countval=0
		for j in i.leans_to:
			sumval+=j.points
			countval+=1
		if countval!=0: # no idea what this state can lead to so far
			i.points=sumval/countval-1 # core

# and voila

# i should write why i do that or that

	def save(self, filename):
		pass # default filename will be /tmp/qnn.tmp.dat

	def load(self, filename):
		pass



class translate: # there will be two of them, one to translate state, one for actions
	pass


#####
# i need a function to define the win state first. the save/load function will include it anyway. remember to save into tmp
# to test, i'll have amonst other Rembmer(self, 99, 101) # that is a loss
# Remember(self, 99, 100) # that is a win. The avg should teach the qnn that from 91 to 99 is a good position, meaning that from 89 can lead to 90 that is eventually a loss

# i should also build an interface, so that i can use sys.argv and parse it
# meaning that if sys.argv[1] = check
# or sys.argv[1] = remember
# or sys.argv[1] = best
# then i do what's written
# note : if i use command line, then i'll use also an interface to translate a state in string into a bool. same for actions possibles, that will be another list of parameters
# that will be another object anyway, but i'll need it soon though

