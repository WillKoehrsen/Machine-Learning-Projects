# Welcome to the eight puzzle solver. 

The eight puzzle solver finds solution paths to the classic eight puzzle
algorithmically using a-star search and local beam search. 

## Author: William Koehrsen
## Date: September 20, 2017

## Requirements

python 3.5.4+

## Files 

**report.pdf** 

Contains full documentation of the project with citations and comprehensive
examples demonstrating the required methods. The report was written as a 
Jupyter Notebook and compiled to html and pdf.


**eight_puzzle.py** 

Contains the Puzzle class with implementations of all 
required methods

**play_puzzle.py** 

Contains a command line parser and file parser for 
interacting with the eight puzzle through the command line or through a series 
of commands in a text file. 

**images/**

Contains R graphs used in the report.

**tests/**

Contains text files used to send sequences of commands to the eight puzzle
from the command line. 

**report.ipynb** 

Original Jupyter Notebook of report. 

## Command line arguments

-setState state_string		

Sets the current state of the puzzle. The format 
for the state_string is "b12 345 678" where each triple is a row of the puzzle and "b" is the blank.

-randomizeState num_moves		

Randomizes the state of the puzzle taking an 
integer number of random moves backwards from the goal state. This ensures 
the resulting puzzle state will have a solution. (Only half of all random
starting states are solvable)

-printState (no arguments) 

This will display the current state of the puzzle
in the format "b12 345 678" where each triple is a row and "b "is the blank 
tile. 

-move action 		

Executes one specified action where the action is  one of ["up", "down", 
"left", "right"]. Allowed moves depend on the state of the puzzle. 

-solveAStar	heuristic		

Solves the puzzle using astar search and
the specified heuristic. The heuristic is one of ["h1", "h2"]. h1 counts the 
number of misplaced tiles. h2 is the sum of the Manhattan distances of all the 
tiles from the goal position.

-solveBeam k		

Solves the puzzle using local beam search with an integer k beam width. A 
wider beam requires more memory, but is more likely to find a solution

-maxNodes max_nodes			

Set a maximum number of nodes to be considered for
the searches. A node is considered when it is generated. If no number is 
specified, the default maximum is 10000 nodes. The search will return 
unsuccessfully if it does not find a solution within the max number of nodes.

-readCommands f.txt 		

Read a series of the above commands from the specified text file. Commands are 
in the same format as from the command line except without the leading hyphen. 

## Additional Methods

-prettyPrintState (no arguments) 		

Displays the state of the puzzle in an aestheic format where 0 represents the 
blank tile. 

-prettyPrintSolution (no arguments)

Displays the solution path of the puzzle in an aesthetic format where 0
represents the blank tile

# Examples

## Command Line Operation

1. Randomize the state of the puzzle with 10 random moves from the goal,
print the state, and try to solve using astar with the h1 heuristic and 2000 max nodes. 

python play_puzzle.py -randomizeState 10 -printState -solveAStar "h1"
-maxNodes 2000

2. Set the state of the puzzle to "312 645 b78"
and solve using astar with the h2 heuristic and 1000 max nodes. Pretty
print the resulting solution. 

python play_puzzle.py -setState "312 645 b78" -solveAStar "h2" -maxNodes 1000 -prettyPrintSolution

3. Randomize the state of the puzzle with 15 random moves from the goal and 
solve using local beam with k = 50 and 5000 max nodes. 

python play_puzzle.py -randomizeState 15 -solveBeam 50 -maxNodes 5000

4. Set the state of the puzzle to "3b5 421 678", and try to
solve using local beam search with k = 100 and 5000 max nodes. Pretty print
the solution. 

python play_puzzle.py -setState "3b5 421 678" -solveBeam 100 -maxNodes 5000 -prettyPrintSolution

## File Reading Operation

1. Randomize the state of the puzzle using 16 random moves from the goal 
and try to solve using astar search with the h2 heuristic and the default 
number of maximum nodes. pretty print the resulting solution path. 

Command line input

python play_puzzle.py -readCommands tests/randomize_solve_astar.txt

"randomize_solve_astar.txt":

randomizeState 16 solveAStar h2 prettyPrintSolution

2. Set state to "3b2 615 748", print the state, and try to solve using 
local beam search with k = 50 and 2000 max nodes. 

Command line input

python play_puzzle.py -readCommands tests/set_state_solve_beam.txt

"set_state_solve_beam.txt":

setState "3b2 615 748" printState solveBeam 50 maxNodes 2000
