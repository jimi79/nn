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
import numpy as np

def init_ai(name): 
	ai=qlv2.Qlearning(27,9) # input is check for alice, and bob, and an action
	ai.verbose=True
	ai.filename="%s_ttt_ql.tmp" % name
	ai.logfilename="%s_ttt_ql.log" % name 
	ai.random=False

	ai.nn_action.min_cpt=1000
	ai.nn_action.min_data=1000
	ai.nn_action.max_data=2000 
	ai.nn_action.online=True
	ai.nn_opponent.nn.params.lambda_=0
	ai.nn_action.nn.params.verbose=True
	ai.nn_action.nn.init_syns([27],27,18) # 2 hidden layers 
	ai.nn_action.nn.params.filename="%s_ttt_nn_action.tmp" % name
	ai.nn_action.nn.try_load_synapses() 
	ai.nn_action.nn.params.max_cpt=2000
	ai.nn_action.nn.params.min_J_cv=0.1
	ai.nn_action.nn.params.check_every_n_steps=100

	ai.nn_opponent.min_cpt=5000
	ai.nn_opponent.min_data=5000
	ai.nn_opponent.max_data=20000 
	ai.nn_opponent.online=True
	ai.nn_opponent.nn.params.lambda_=0
	ai.nn_opponent.nn.params.verbose=True
	ai.nn_opponent.nn.init_syns([],18,18) # 2 hidden layers 
	ai.nn_opponent.nn.params.filename="%s_ttt_nn_opponent.tmp" % name
	ai.nn_opponent.nn.try_load_synapses() 
	ai.nn_opponent.nn.params.min_J_cv=0.2
	ai.nn_opponent.nn.check_every_n_steps=100 

	ai.max_action=9
	ai.set_name(name)
	ai.try_load()
	return ai



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


alice=init_ai("alice")
bob=init_ai("bob") 
all_actions=list(range(1,11)) 

def print_game(alice, bob):
	c=["X" if alice[i]==1 else "O" if bob[i]==1 else " " for i in range(9)]
	print("%s│%s│%s" % (c[0],c[1],c[2]))
	print("─┼─┼─")
	print("%s│%s│%s" % (c[3],c[4],c[5]))
	print("─┼─┼─")
	print("%s│%s│%s" % (c[6],c[7],c[8]))

def print_history(array, winner=None):
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
			if not winner is None:
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


def play_AI(ai, ai2, board, board2, verbose=True): 
	available_actions=get_available_actions(list_or(board, board2)) 
	if verbose:
		print("I ask %s to play" % ai.name)
	if verbose:
		print("available actions are %s" % available_actions)
	action=ai.pick_action(board+board2, available_actions) 
	board[action]=1 
	win=is_win(board)

	tie=False
	if not win:
		if sum(board)+sum(board2)==9:
			tie=True

	points_ai=None 
	points_ai2=None 
	if win:
		points_ai=1000
		points_ai2=-1000
	if tie:
		points_ai=-100
		points_ai2=-100 # a tie isn't good

	ai.learn_action(board+board2)
	ai2.learn_opponent(board2+board)

	#def learn_points(self, state, points, win):
	if points_ai != None: 
		ai.learn_points(board+board2, points_ai, win)
	if points_ai2 != None: 
		ai2.learn_points(board2+board, points_ai2, False)

	return tie, win, board, board2


def play(verbose):
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
		tie, win, board_alice, board_bob=play_AI(alice, bob, board_alice, board_bob, verbose)
		if win:
			winner="alice"
		if verbose:
			print_game(board_alice, board_bob) 

		h=[a+b*2 for a,b in zip(board_alice, board_bob)]
		history.append(h)

		if (not tie) and (not win):
			tie, win, board_bob, board_alice=play_AI(bob, alice, board_bob, board_alice, verbose)
			if win:
				winner="bob"
			if verbose:
				print_game(board_alice, board_bob) 

			h=[a+b*2 for a,b in zip(board_alice, board_bob)]
			history.append(h)
		if tie:
			winner="tie"

		# we learn the first ai first nn what happens
# we learn the other ai other ai what happens, if we got infos about it

	return winner,history

def loop(count, verbose_games=None, verbose_detail=None, verbose_stats=None, force=None, verbose_training=None):
	score_bob=0
	score_alice=0
	tie=0
	stats=0
	duration=0
	if verbose_stats==None:
		verbose_stats=1000
	if verbose_games is None:
		verbose_games=count<101
	if verbose_detail is None:
		verbose_detail=count<11
	if verbose_training is None:
		verbose_training=count<101


	alice.set_verbose(verbose_detail)
	bob.set_verbose(verbose_detail)

	alice.nn_action.verbose=verbose_training
	alice.nn_opponent.verbose=verbose_training
	bob.nn_action.verbose=verbose_training
	bob.nn_opponent.verbose=verbose_training

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
	
def print_game_from_int(integer):
	a=qlv2.integer_to_array(integer)
	b=a[0:9]
	c=a[9:19]
	print_game(b,c)

def print_game_from_dataset(dataset, idx):
	X=dataset.X[idx]
	y=dataset.y[idx]
	a=X[0:9]
	b=X[9:18]
	act=X[18:]
	c=y[0:9]
	d=y[9:18]
	print_game(a,b)
	print("Action %d :" % np.where(act)[0][0])
	print_game(c,d)


def save():
	alice.save()
	bob.save()

def load():
	if alice.try_load():
		print("alice loaded")
	if bob.try_load():
		print("bob loaded")

def print_ai_guess_from_int(integer, action):
	act=copy.copy(zeros)
	act[actions]=1
	print_game_from_int(integer)
	res=alice.nn.FP(np.array(qlv2.integer_to_array(136400)+act))>0.5
	res=qlv2.array_to_integer(res)
	print("res=%d" % res)
	print_game_from_int(res)


load()

# the ql that is used. i make it global so i can study it easily
if len(sys.argv)>1:
	if sys.argv[1]=='test':
		loop(20000, verbose_detail=True, verbose_games=True, force=True)
