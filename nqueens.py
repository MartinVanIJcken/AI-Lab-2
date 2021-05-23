import sys
import random
import math

MAXQ = 100


def in_conflict(column, row, other_column, other_row):
    """
    Checks if two locations are in conflict with each other.
    :param column: Column of queen 1.
    :param row: Row of queen 1.
    :param other_column: Column of queen 2.
    :param other_row: Row of queen 2.
    :return: True if the queens are in conflict, else False.
    """
    if column == other_column:
        return True  # Same column
    if row == other_row:
        return True  # Same row
    if abs(column - other_column) == abs(row - other_row):
        return True  # Diagonal

    return False



def in_conflict_with_another_queen(row, column, board):
    """
    Checks if the given row and column correspond to a queen that is in conflict with another queen.
    :param row: Row of the queen to be checked.
    :param column: Column of the queen to be checked.
    :param board: Board with all the queens.
    :return: True if the queen is in conflict, else False.
    """
    for other_column, other_row in enumerate(board):
        if in_conflict(column, row, other_column, other_row):
            if row != other_row or column != other_column:
                return True
    return False


def count_conflicts(board):
    """
    Counts the number of queens in conflict with each other.
    :param board: The board with all the queens on it.
    :return: The number of conflicts.
    """
    cnt = 0

    for queen in range(0, len(board)):
        for other_queen in range(queen+1, len(board)):
            if in_conflict(queen, board[queen], other_queen, board[other_queen]):
                cnt += 1

    return cnt


def evaluate_state(board):
    """
    Evaluation function. The maximal number of queens in conflict can be 1 + 2 + 3 + 4 + .. +
    (nquees-1) = (nqueens-1)*nqueens/2. Since we want to do ascending local searches, the evaluation function returns
    (nqueens-1)*nqueens/2 - countConflicts().

    :param board: list/array representation of columns and the row of the queen on that column
    :return: evaluation score
    """
    return (len(board)-1)*len(board)/2 - count_conflicts(board)


def print_board(board):
    """
    Prints the board in a human readable format in the terminal.
    :param board: The board with all the queens.
    """
    print("\n")

    for row in range(len(board)):
        line = ''
        for column in range(len(board)):
            if board[column] == row:
                line += 'Q' if in_conflict_with_another_queen(row, column, board) else 'q'
            else:
                line += '.'
        print(line)


def init_board(nqueens):
    """
    :param nqueens integer for the number of queens on the board
    :returns list/array representation of columns and the row of the queen on that column
    """

    board = []

    for column in range(nqueens):
        board.append(random.randint(0, nqueens-1))

    return board


"""
------------------ Do not change the code above! ------------------
"""


def random_search(board):
    """
    This function is an example and not an efficient solution to the nqueens problem. What it essentially does is flip
    over the board and put all the queens on a random position.
    :param board: list/array representation of columns and the row of the queen on that column
    """

    i = 0
    optimum = (len(board) - 1) * len(board) / 2

    while evaluate_state(board) != optimum:
        i += 1
##              print('iteration ' + str(i) + ': evaluation = ' + str(evaluate_state(board)))
        if i == 1000:  # Give up after 1000 tries.
            break

        for column, row in enumerate(board):  # For each column, place the queen in a random row
            board[column] = random.randint(0, len(board)-1)


    return board

def hill_climbing(board):
    """
    This function implements hill climbing with random restarts.
    :param board: list/array representation of columns and the row of the queen on that column
    :return: None
    """
    optimum = (len(board) - 1) * len(board) / 2
    dead_ends = []
    # the improvement: random restarts  in this list we track what pieces we have
    # already tried and failed to find a better alternative for

    
    for i in range(10000):
##              print('iteration ' + str(i) + ': evaluation = ' + str(evaluate_state(board)))

        if evaluate_state(board) == optimum: # stop when an optimum has been found
            break
        
        worst_queens = [(0, 0)]
        # by definition 0 is the best it could be thus any imperfect queen will replace this
        
        for queen in range(0, len(board)):
            if queen in dead_ends:
                continue
            
            badness = 0
            
            for other_queen in range(0, len(board)):
                if queen == other_queen:
                    continue
                
                if in_conflict(queen, board[queen], other_queen, board[other_queen]):
                    badness += 1

            # determine whether this is worse or as bad as the current worst
            if badness > worst_queens[0][1]:
                worst_queens = [(queen, badness)]
            elif badness == worst_queens[0][1]:
                worst_queens.append((queen, badness))
                
        worst_queen = random.choice(worst_queens)

        # do some variable renaming that makes future code easier
        best_alternatives = [(board[worst_queen[0]], worst_queen[1])]
        worst_queen, _ = worst_queen
        current_location = board[worst_queen]
        
        for alternative in range(0, len(board)):
            
            badness = 0

            for other_queen in range(0, len(board)):
                if alternative == other_queen:
                    continue
                
                if in_conflict(worst_queen, alternative, other_queen, board[other_queen]):
                    badness += 1

            # determine whether this is better or as good as the current best
            if badness < best_alternatives[0][1]:
                best_alternatives = [(alternative, badness)]
            elif badness == best_alternatives[0][1]:
                best_alternatives.append((alternative, badness))

        best_alternatives = [x[0] for  x in best_alternatives]
        
        if current_location in best_alternatives:
            # the best alternative is as bad as the current position => this is a dead end
            dead_ends.append(worst_queen)

            if len(dead_ends) > len(board)/4: # a cutoff point chosen based on best performance
                # if we failed to find an alternative for all N states, we do a restart
##              print("Random restart")
                for column, row in enumerate(board):  # For each column, place the queen in a random row
                    board[column] = random.randint(0, len(board)-1)
                    dead_ends = []
        else:
            board[worst_queen] = random.choice(best_alternatives)
            dead_ends = [] # once a move is done, old dead ends might be new highways

    return board

def _time_to_temperature(t, T0=1, decayrate=0.995):
    return T0*decayrate**t # changed from linear to exponential now it works rather nicely
##  return 1-0.0004*t

def simulated_annealing(board, ttoT):
    """
    This function implements simulated annealing, every iteration it randomly moves one of the
    queens to a random row in the same column
    :param board:
    :return:
    """
    optimum = (len(board) - 1) * len(board) / 2
    
    t = 0
    T = ttoT(t)
    
    while (T := ttoT(t)) > 0.0001 and evaluate_state(board) != optimum:
        print(f"iteration {t}: evaluation = {evaluate_state(board)}, T={T}")
        t += 1


        sucessor = board.copy()
        sucessor[random.randint(0, len(board)-1)] = random.randint(0, len(board)-1)
        
        value_difference = evaluate_state(sucessor)-evaluate_state(board)

        if value_difference > 0:
            board = sucessor
        else:
            if random.random() < 2.71828**(value_difference/T):
                board = sucessor

    return board


def genetic_algorithm(boards, population_size):
    '''
    Fitness function is the evaluate_board function.
    '''
    # set things up
    population = boards
    n_queens = len(boards[0])
    halfway = n_queens//2
    i=0
    mutation_prob = 2**(-i/10)
    optimum = (n_queens - 1) * n_queens / 2
    fitness_scores = [evaluate_state(pop) for pop in population]
    
    # run the loop
    while max(fitness_scores) != optimum:
        i += 1

        new_population = []

        # set up list of fitness scores
        
        
        for n in range(population_size):
            # select random elements from population with bias for fitness
            x, y = random.choices(population, fitness_scores, k=2)
            # create their child by splicing the lists
            child = x[:halfway]+y[halfway:]
            # small chance to mutate
            if random.random() < mutation_prob:
                child[random.randint(0, len(child)-1)] = random.randint(1, n_queens)
            # add child to the population
            new_population.append(child)
        population = new_population
        fitness_scores = [evaluate_state(pop) for pop in population]
        if i == 10000:  # Give up after 10 000 tries.
##                      print("Gave up.")
##                      print_board(population[fitness_scores.index(max(fitness_scores))])
            break

    return population[fitness_scores.index(max(fitness_scores))]

def test(alg):
    startingpoint = 4
    cycles = 5
    N = 10
    results = []
    for x in range(N):
        tmp_results = 0
        n_queens = startingpoint * (2 ** x)
        optimum = n_queens * (n_queens-1)/2

        for i in range(0, N):
##            board = init_board(n_queens)
            population_size = 8
            boards = [init_board(n_queens) for i in range(population_size)]
            
            if evaluate_state(alg(boards, population_size)) == optimum:
                tmp_results += 1
    
        print(n_queens, tmp_results / N)

def main():
    """
    Main function that will parse input and call the appropriate algorithm. You do not need to understand everything
    here!
    """

    try:
        if len(sys.argv) != 2:
            raise ValueError

        n_queens = int(sys.argv[1])
        if n_queens < 1 or n_queens > MAXQ:
            raise ValueError

    except ValueError:
        print('Usage: python n_queens.py NUMBER')
        return False

    print('Which algorithm to use?')
    algorithm = input('1: random, 2: hill-climbing, 3: simulated annealing 4: genetic algorithm\n')

    try:
        algorithm = int(algorithm)

        if algorithm not in range(1, 5):
            raise ValueError

    except ValueError:
        print('Please input a number in the given range!')
        return False
    if algorithm != 4:
        board = init_board(n_queens)
        print('Initial board: \n')
        print_board(board)
    else:
        # because we start with multiple boards, we need to do this stuff
        population_size = 8
        boards = [init_board(n_queens) for i in range(population_size)]

        print('Initial boards: \n')
        for i in range(population_size):
            print_board(boards[i])

    if algorithm == 1:
        result = random_search(board)
    elif algorithm == 2:
        result = hill_climbing(board)
    elif algorithm == 3:
        result = simulated_annealing(board, _time_to_temperature)
    elif algorithm == 4:

        
        
        result = genetic_algorithm(boards, population_size)

    optimum = (n_queens - 1) * n_queens / 2
    if evaluate_state(result) == optimum:
        print('Solved puzzle!')

    print('Final state is:')
    print_board(result)

        
        

# This line is the starting point of the program.
if __name__ == "__main__":
    main()
