#!/usr/bin/python3

#ok, so, there should be two functions in my class. one that ask 'what do you want to do', second that says 'well, you did that and that happened'. one extra parameter to say 'and you win'. Because it should win at the end

import numpy as np 
import random

class Qlearning:

	class State:
		def __init__(self, ident, points=None):
			self.ident=ident # identification of that state
			self.leads_to=[] # array of idents of states that it can lead to (i prefer that that storing the pointer)
			if points==None: 
				points=0
			self.points=points # value of that state 

	def __init__(self):
		self.states=[] # will be filled with states

	def __test(self): # that is private :)
		print("private") 

	def find(self, ident):
		for i in self.states:
			if i.ident == ident:
				return i # return a state
				break
		return None

	def define(self, state, points=None): # can be used to add the win value 
		print("blah")
		i=self.find(state)
		if i!=None:
			if points!=None:
				i.points = points
		else: 
			self.states.append(self.State(state, points))

	def query(self, state, possible_actions):
		self.define(state)
		return self.best_possible_action(possible_actions)
# return here the best possible_action

	def best_possible_action(self, possible_actions):
		return possible_actions[random.randrange(0,len(possible_actions))] # i should use a NN for that, based on the score


	def remember(self, old_state, new_state, action):
		self.define(new_state) # we don't know if that new state is good now
		i=self.find(old_state) # shoulod exists

		if i.leads_to.count(new_state)==0:
			i.leads_to.append(new_state)

		sumval=0
		countval=0
		for ii in i.leads_to:
			j=self.find(ii) 
			sumval+=j.points
			countval+=1

		if countval!=0: # no idea what this state can lead to so far
			i.points=sumval/countval-1 # core # ok, avereage is good if the outcome is random. otherwise it shouldn't be
			# actually, it depends of the action we picked. if the action leads to two different outcomes, then we average
# but we take the max for both actions
# so i may need a matrice after all, that says 'action blah and status blah leads to blah2, or blah3, depends' 

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



#todo
# what if going from state1 to state2 costs more than going from state1 to state3, if both lead to state4 ? I need an optionnal matrice of costs too. so that when i do -1, it could be another value
# but that may be for later.
