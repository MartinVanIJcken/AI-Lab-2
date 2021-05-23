import sys


def max_value(state):
	max = -100000000000

	if state == 1:
		return -1

	for move in range(1, 4):
		if state-move > 0:
			m = min_value(state-move)
			max = m if m > max else max

	return max


def min_value(state):
	min = 10000000000000

	if state == 1:
		return 1

	for move in range(1, 4):
		if state-move > 0:
			m = max_value(state-move)
			min = m if m < min else min

	return min


def minimax_decision(state, turn):
	best_move = None

	if turn == 0:  # MAX' turn
		max = -100000000000

		for move in range(1, 4):
			if state - move > 0:
				m = min_value(state - move)
				if m > max:
					max = m
					best_move = move

	else:
		min = 10000000000000

		for move in range(1, 4):
			if state - move > 0:
				m = max_value(state-move)
				if m < min:
					min = m
					best_move = move

	return best_move

#def negamax(node, depth, utility):
#    if node == 0:
#        return utility
#    if depth == 0:
#        print('No result.')
#        return 0
#    value = -1e6
#    for child in [node - n for n in range(1, min(node, 3)+1)]:
#        value = max(value, negamax(child, depth - 1, -1*utility))
#    print(node, value, depth, utility)
#    return -value
def negamax(node, utility, current_depth, max_depth=50):
	# utility = 1 means it's max's turn
	utility = -utility
	if node == 0:
		#print(node, utility, current_depth)
		return utility
	
	if current_depth == max_depth:
		print('No result.')
		return 0
	value = -utility
	for child in [node - n for n in range(1, min(node, 3)+1)]:
		if child >= 0:
			if utility == 1:
				value = max(value, negamax(child, utility, current_depth + 1))
			else:
				value = min(value, negamax(child, utility, current_depth + 1))
	#print(node, value, current_depth)
	return value


def play_nim(state):
	turn = 0

	while state != 1:
		move = minimax_decision(state, turn)
	 #   print(str(state) + ": " + ("MAX" if not turn else "MIN") + " takes " + str(move))

		state -= move
		turn = 1 - turn

	#print("1: " + ("MAX" if not turn else "MIN") + " looses")
	if turn:
		return 1
	else:
		return -1



def main():
	"""
	Main function that will parse input and call the appropriate algorithm. You do not need to understand everything
	here!
	"""

	try:
		if len(sys.argv) != 2:
			raise ValueError

		state = int(sys.argv[1])
		if state < 1 or state > 100:
			raise ValueError

	except ValueError:
		print('Usage: python nim.py NUMBER')
		return False
	
	errors = 0
	for state_num in range(1, state):
		print('Starting amount: ' + str(state_num))
		print(play_nim(state_num))
		print(negamax(state_num, -1, 0))
		print('\n')

		errors += (play_nim(state_num) - negamax(state_num, -1, 0))**2

	print('\nTotal amount of errors: ' + str(errors))


if __name__ == '__main__':
	main()
