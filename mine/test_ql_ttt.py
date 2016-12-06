#!/usr/bin/python3

# first : change the ql thiny so that action is an array, and not an integer counting the possible actions



# note : the representation of the state is an array, but as well as the state stored to know if it wins or not. I think i will use a string, even if it could be a binary.
# I just have to pay attention on how it will works inside the ql, because so far it was just an array indexed by an integer


# we got two arrays, which having the list of 9 cells. 
# one array for alice. if 1, it means there is a cross here
# one array for bob. if 1, it means there is a round here

# random start

# o
#    Bot │
# ─┼─

# and i need a procedure to check if someone wins.
# for that, i will generate the list that are a win, with a procedure
# then i will try to find the guy's list in the list of list. 

import copy
import importlib
import pdb
import qlv2
import sys



zeros=[0,0,0,0,0,0,0,0,0]
wins=[]
wins.append([1,1,1,0,0,0,0,0,0])
wins.append([0,0,0,1,1,1,0,0,0])
wins.append([0,0,0,0,0,0,1,1,1])
wins.append([1,0,0,1,0,0,1,0,0])
wins.append([0,1,0,0,1,0,0,1,0])
wins.append([0,0,1,0,0,1,0,0,1])
wins.append([1,0,0,0,1,0,0,0,1])
wins.append([0,0,1,0,1,0,1,0,0])

def print_game(alice, bob):
	c=["X" if alice[i]==1 else "O" if bob[i]==1 else " " for i in range(9)]
	print("%s│%s│%s" % (c[0],c[1],c[2]))
	print("─┼─┼─")
	print("%s│%s│%s" % (c[3],c[4],c[5]))
	print("─┼─┼─")
	print("%s│%s│%s" % (c[6],c[7],c[8]))

def print_history(array, winner):
	for i in range(3):
		s=""
		s2=""
		for h in array:
			a=h[i*3:(i*3)+3]
			b=["X" if i==1 else "O" if i==2 else "?" if i!=0 else " " for i in a]
			s+="%s│%s│%s " % (b[0], b[1], b[2])
			s2+="─┼─┼─ "
		print(s)
		if i==1:
			s2+=" winner:%s" % winner
		if i < 2:
			print(s2)

def list_or(list1,list2):
	return [a or b for a,b in zip(list1,list2)]

def get_available_actions(busy):
	l=[1-i for i in busy]
	return [i for i in range(len(l)) if l[i]==1]

def is_win(list_):
	for i in wins:
		if sum(i)==sum([a*b for a,b in zip(list_,i)]):
			return True
	return False



# alice plays. if it wins, then it remember that the status it has is winnning
# if then bob wins, alice should remember nothg. 
# that is more like q learning, because you don't loose by your own action. You lose because of other's action


def play_AI(ai, board, board2, verbose=True): 
	status=board+board2
	available_actions=get_available_actions(list_or(board, board2))

	if verbose:
		print("I ask %s to play" % ai.name)
	if verbose:
		print("available actions are %s" % available_actions)
	action=ai.pick_action(status, available_actions) 
	board[action]=1 
	win=is_win(board)

	tie=False
	if not win:
		if sum(board)+sum(board2)==9:
			tie=True

	points_ai=None 
	if win:
		points_ai=1000
	if tie:
		points_ai=100
	ai.learn(board+board2, points_ai)  # we learn the AI what happened, which is bad bad bad bad
# i mean, the nn has to go to the good situation, which is the next one, not the next next one
# but the status that is good, is the next next one
# i am mixing up the NN that can play with the NN that can foresee what the play will be

	return tie, win, board, board2


def play(verbose):
	global alice
	global bob
	board_alice=[0 for i in range(9)]
	board_bob=copy.copy(board_alice)
	alice.restart()
	bob.restart() 
	win=False
	tie=False
	cpt=0
	history=[]
	winner=""
	while (not win) and (not tie):
		cpt+=1
		if cpt>22:
			raise Exception("loop")
		tie, win, board_alice, board_bob=play_AI(alice, board_alice, board_bob, verbose)
		if win:
			winner="alice"
		if verbose:
			print_game(board_alice, board_bob) 

		h=[a+b*2 for a,b in zip(board_alice, board_bob)]
		history.append(h)

		if (not tie) and (not win):
			tie, win, board_bob, board_alice=play_AI(bob, board_bob, board_alice, verbose)
			if win:
				winner="bob"
			if verbose:
				print_game(board_alice, board_bob) 

			h=[a+b*2 for a,b in zip(board_alice, board_bob)]
			history.append(h)
		if tie:
			winner="tie"

	return winner,history

def loop(count, verbose_games=False, verbose_detail=False, verbose_stats=False, force=False, verbose_nn=False):
	global alice
	global bob
	score_bob=0
	score_alice=0
	tie=0
	stats=0
	duration=0
	if verbose_stats==True:
		verbose_stats=100 # default value 

	alice.verbose=verbose_detail
	bob.verbose=verbose_detail
	alice.nn.nn.verbose=verbose_nn
	bob.nn.nn.verbose=verbose_nn

	if count>100:
		if verbose_detail:
			if not force:
				print("are you sure u want detail for more than 100 iterations ? use force=True then")
				return
		if count > 1000:
			if verbose_games:
				if not force:
					print("are you sure u want the history for more than 1000 iterations ? use force=True then")
					return

	for i in range(count):
		stats+=1
		winner,history=play(verbose_detail)
		duration+=len(history)
		if winner=="tie":
			tie+=1
		if winner=="alice":
			score_alice+=1
		if winner=="bob":
			score_bob+=1

		if verbose_games:
			print_history(history,winner)

		if stats==verbose_stats:
			if verbose_stats:
				print("For the last %d games : alice %.2f, Bob %.2f, Tie %.2f, duration game %.2f, count %d" % (verbose_stats, score_alice/stats*100, score_bob/stats*100, tie/stats*100, duration/stats, i+1))
			duration=0
			score_alice=0
			score_bob=0
			tie=0
			
			stats=0
	
def init_ai(name): 
	ai=qlv2.Qlearning(27,9) # input is check for alice, and bob, and an action
	ai.nn.init_syns([27],27,18) # 2 hidden layers 
	ai.min_data_to_train=10000 # default value
	ai.max_data_to_train=100000 # default value
	ai.min_cpt_since_last_train=10000 # don't train every time
	ai.cpt_since_last_train=0
	ai.verbose=True
	ai.filename="%s_ttt.tmp" % name
	ai.nn.verbose=True
	ai.nn.nn.filename="%s_ttt_nn.tmp" % name
	ai.nn.try_load_synapses() 
	ai.nn.nn.max_cpt=2000
	ai.nn.min_J_cv=0.05
	ai.nn.nn.check_every_n_steps=100
	ai.max_action=9
	ai.name=name
	ai.try_load()
	return ai

def print_game_from_int(integer):
	a=qlv2.integer_to_array(integer)
	b=a[0:9]
	c=a[9:19]
	print(b)
	print(c)
	print_game(b,c)

def save():
	alice.save()
	bob.save()

def load():
	alice.load()
	bob.load()



# the ql that is used. i make it global so i can study it easily
alice=init_ai("alice")
bob=init_ai("bob") 
all_actions=list(range(1,11))

if len(sys.argv)>1:
	if sys.argv[1]=='test':
		loop(20000, verbose_detail=True, verbose_games=True, force=True)
