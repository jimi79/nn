#!/usr/bin/python3

#ok, so, there should be two functions in my class. one that ask 'what do you want to do', second that says 'well, you did that and that happened'. one extra parameter to say 'and you win'. Because it should win at the end

import numpy as np 

class Qlearning:
	def __init__(self):
		Qmatrice = np.zeros(0)
		States = [] # will be a list of list, to find the index of a given state
		Actions = [] # will return an indice too
	
	def Query(self, state, possible_actions):
# here, we update all combinations with all new possible states, with the max. but only if it's possible. so we don't
# meaning we update only the state that will work, meaning we don't know, meaning we don't

		pass

	def Remember(self, old_state, new_state, action):
# first, if we win, then we write 100 in the new old_state new_state combination
# if we don't win, then we update the path we took, meaning the old_state / new_state cell with the gamma * max(new_state) row (meaning all possible new states + 1)
# we do it only here, because we know that going from old to new is possible
# it's here that we'll reinforce old_state thing
		pass

# and voila

# i should write why i do that or that
