import argparse
import fileinput
from eight_puzzle import Puzzle

if __name__=="__main__":
	# Initialize the puzzle
	puzzle = Puzzle()

	# Create a command line parser and add appropriate arguments
	parser = argparse.ArgumentParser(description="Interact with the Eight Puzzle")
	parser.add_argument('-setState', help='Enter a state for the puzzle in format "b12 345 678"', metavar = '')
	parser.add_argument('-randomizeState', help='Enter an integer number of random steps to take backwards from goal', type=int, metavar = '')
	parser.add_argument('-printState', action="store_true", help='Display current state of puzzle')
	parser.add_argument('-move', help='Move the blank one tile in the specified direction', metavar = '')
	parser.add_argument("-solveAStar", help='Solve the eight puzzle using the specified heuristic', metavar = '')
	parser.add_argument("-solveBeam", help='Solve the eight puzzle using the local beam search with the specified number of state',type=int, metavar = '')
	parser.add_argument("-maxNodes", help='Specify a maximum number of nodes to explore while searching', default=None, type=int, metavar = '')
	parser.add_argument("-prettyPrintState", help='Display the current state in an aesthically pleasing manner', action="store_true")
	parser.add_argument("-prettyPrintSolution", help='Display the solution path in an aesthically pleasing manner', action="store_true")
	parser.add_argument("-readCommands", help="Read and execute a series of commands from a text file where commands are specified without dashes")

	# args is all the arguments provided
	args = parser.parse_args()

	# Series of if statements to deal with arguments
	if args.setState:
		puzzle.set_state(args.setState)

	if args.randomizeState:
		puzzle.randomize_state(args.randomizeState)

	if args.printState:
		print("Current puzzle state:")
		puzzle.print_state()

	if args.move:
		puzzle.state = puzzle.move(puzzle.state, args.move)	

	if args.solveAStar:
		if args.maxNodes:
			puzzle.a_star(heuristic = args.solveAStar, max_nodes = args.maxNodes)
		else:
			puzzle.a_star(heuristic = args.solveAStar)

	if args.solveBeam:
		if args.maxNodes:
			puzzle.local_beam(k = args.solveBeam, max_nodes = args.maxNodes)
		else:
			puzzle.local_beam(k = args.solveBeam)

	

	if args.prettyPrintState:
		puzzle.pretty_print_state(puzzle.state)

	if args.prettyPrintSolution:
		puzzle.success(puzzle.expanded_nodes, puzzle.num_nodes_generated, print_solution=False) 
		puzzle.pretty_print_solution(puzzle.solution_path)

	# If a text file is provided, need to iterate through commands in file
	if args.readCommands:

		# Open the text file and read line by line
		with open(args.readCommands, "r") as f:
			for line in f:
				# Split the commands into a list on spaces
				arguments = line.split(" ")

				# Strip out the newline character from the last argument
				arguments[-1] = arguments[-1].strip()

				# Iterate through the arguments
				for position, argument in enumerate(arguments):
					if argument == "setState":
						puzzle.set_state(" ".join(arguments[position + 1: position + 4]).strip('\"'))

					elif argument == "randomizeState":
						puzzle.randomize_state(int(arguments[position + 1]))

					elif argument == "move":
						puzzle.state = puzzle.move(puzzle.state, arguments[position + 1].strip('\"'))

					elif argument == "solveAStar":
						heuristic = arguments[position + 1]
						try: 
							max_nodes_index = arguments.index("maxNodes")
							max_nodes = int(arguments[max_nodes_index])
							puzzle.a_star(heuristic = heuristic, max_nodes = max_nodes)
						except:
							puzzle.a_star(heuristic = heuristic)

					elif argument == "solveBeam":
						k = int(arguments[position + 1])
						try: 
							max_nodes_index = arguments.index("maxNodes")
							max_nodes = int(arguments[max_nodes_index])
							puzzle.local_beam(k = k, max_nodes = max_nodes)
						except:
							puzzle.local_beam(k = k)

					elif argument == "printState":
						print("Current puzzle state:")
						puzzle.print_state()

					elif argument == "prettyPrintState":
						puzzle.pretty_print_state(puzzle.state)

					elif argument == "prettyPrintSolution":
						puzzle.success(puzzle.expanded_nodes, puzzle.num_nodes_generated, print_solution=False)
						puzzle.pretty_print_solution(puzzle.solution_path)