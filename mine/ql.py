#!/usr/bin/python3


# first, we got each state that got a weight
# then, each state can lead to another state, depending on a choice, and a ratio of chances

# ok, first : cost



class State:
	def __init__(self, ident, points=None):
		self.ident=ident
		if points==None:
			points=0 #not good, not bad
		self.points=points
		self.possibles_actions=[] # array of indexes of actions here. 

class Action: # action going west in city != action going west in forest. So an action is basically an action in a given state
	def __init__(self):
		self.leads_to=[] #one action can lead to several outcomes. Here, there is also an index
		self.leads_to_count=[] # array of count of each time the outcome happened for that given action
		self.tot_count # numer of times that action was called
		self.cost=1 # cost value, default is 1

#core : avereage of each action, then max of all possibles actions, minus cost. Cost is initialized by the 'remember' function (default 1, cannot be 0 i guess, otherwise there is no reason to move.)
	

class Qlearning():
	def __init__(self):
		self.states=[]
		self.actions=[]

	def find_state(self, ident):
		for i in range(len(self.states)):
			if self.states[i].ident==ident:
				return i
				break
		return -1
	
	def find_action(self, ident): # not sure i will need it here though
		for i in range(len(self.actions)):
			if self.actions[i].ident==ident:
				return i
				break
		return -1

	def define_actions(self, ident, actions, weight=None): # weight not handled yet. see note1
		act=self.find_action(ident)
		if act==None:
			act=Action()
		act.leads_to=actions # i except a list here

	def define_points(self, ident, points=None): 
		




#note1 : weight will be a list of weight for each action, that i will use to substract to the value if i use it, so i can tell which is better, accounting that information
