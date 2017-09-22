import random 
import copy

class Puzzle():
    # Eight puzzle representation with methods for solving

    def __init__(self):
        # Set initial state as the goal state
        random.seed(42)
        self.goal_state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.state = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        
    def goal_check(self, state):
        # Check to see if the given state is the goal state
        return state == self.goal_state
    
    def set_goal_state(self):
        # Set the state of the puzzle to the goal state
        self.state = copy.deepcopy(self.goal_state)
    
    def get_available_actions(self, state):
        # Returns list of available moves and location of blank depending on puzzle state

        #Find the blank tile
        for row_index, row in enumerate(state):
            for col_index, element in enumerate(row):
                if element == 0:
                    blank_row = row_index
                    blank_column = col_index
        
        # Set available moves to empty list
        available_actions = []
        
        # Find available moves, probably a more efficient method exists
        if blank_row == 0:
            available_actions.append("down")
        elif blank_row == 1:
            available_actions.extend(("up", "down"))
        elif blank_row == 2:
            available_actions.append("up")
            
        if blank_column == 0:
            available_actions.append("right")
        elif blank_column == 1:
            available_actions.extend(("left", "right"))
        elif blank_column == 2:
            available_actions.append("left")
            
        # Randomly shuffle the actions to remove ordering bias
        random.shuffle(available_actions)

        return available_actions, blank_row, blank_column
    
    def set_state(self, state_string):
        # Set the state of the puzzle to a string in format "b12 345 678"

        # Check string for correct length
        if len(state_string) != 11:
            print("String Length is not correct!")

        # Keep track of elements that have been added to board
        added_elements = []

        # Enumerate through all the positions in the string
        for row_index, row in enumerate(state_string.split(" ")): 
            for col_index, element in enumerate(row):
                # Check to make sure invalid character not in string
                if element not in ['b', '1', '2', '3', '4', '5', '6', '7', '8']:
                    print("Invalid character in state:", element)
                    break
                else:
                    if element == "b":
                        # Check to see if blank has been added twice
                        if element in added_elements:
                            print("The blank was added twice")
                            break

                        # Set the blank tile to a 0 on the puzzle
                        else:
                            self.state[row_index][col_index] = 0
                            added_elements.append("b")
                    else:
                        # Check to see if tile has already been added to board
                        if int(element) in added_elements:
                            print("Tile {} has been added twice".format(element))
                            break

                        else:
                            # Set the correct tile on the board
                            self.state[row_index][col_index] = int(element)
                            added_elements.append(int(element))
      
    def randomize_state(self, n):
        # Take a random series of moves backwards from the goal state
        # Sets the current state of the puzzle to a state thta is guaranteed to be solvable

        current_state = (self.goal_state)
        
        # Iterate through the number of moves
        for i in range(n):
            available_actions, _, _ = self.get_available_actions(current_state)
            random_move = random.choice(available_actions)
            current_state = self.move(current_state, random_move)
          
        # Set the state of the puzzle to the random state  
        self.state = current_state
            
    def move(self, state, action):
        # Move the blank in the specified direction
        # Returns the new state resulting from the move
        available_actions, blank_row, blank_column = self.get_available_actions(state)

        new_state = copy.deepcopy(state)

        # Check to make sure action is allowed given board state
        if action not in available_actions:
            print("Move not allowed\nAllowed moves:", available_actions)
            return False

        # Execute the move as a series of if statements, probably not the most efficient method
        else:
            if action == "down":
                tile_to_move = state[blank_row + 1][blank_column]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row + 1][blank_column] = 0
            elif action == "up":
                tile_to_move = state[blank_row - 1][blank_column]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row - 1][blank_column] = 0
            elif action == "right":
                tile_to_move = state[blank_row][blank_column + 1]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row][blank_column + 1] = 0
            elif action == "left":
                tile_to_move = state[blank_row][blank_column - 1]
                new_state[blank_row][blank_column] = tile_to_move
                new_state[blank_row][blank_column -1] = 0
                
        return new_state


    def print_state(self):
        # Display the state of the board in the format "b12 345 678"
        str_state = []

        # Iterate through all the tiles
        for row in self.state:
            for element in row:
                if element == 0:
                    str_state.append("b")
                else:
                    str_state.append(str(element))
        
        # Print out the resulting state       
        print("".join(str_state[0:3]), "".join(str_state[3:6]), "".join(str_state[6:9]))

    def pretty_print_state(self, state):
        print("\nCurrent State")
        for row in (state):
            print("-" * 13)
            print("| {} | {} | {} |".format(*row))

    def pretty_print_solution(self, solution_path):
        # Display the solution path in an aesthically pleasing manner
        try:
            # Solution path is in reverse order
            for depth, state in enumerate(solution_path[::-1]):
                if depth == 0:
                    print("\nStarting State")

                elif depth == (len(solution_path) - 2):
                    print("\nGOAL!!!!!!!!!")
                    for row_num, row in enumerate(state[0]):
                        print("-" * 13)
                        print("| {} | {} | {} |".format(*row))

                    print("\n")
                    break
                else:
                    print("\nDepth:", depth)
                for row_num, row in enumerate(state[0]):
                    print("-" * 13)
                    print("| {} | {} | {} |".format(*row))
        except:
            print("No Solution Found")
            
        
    def calculate_h1_heuristic(self, state):
        # Calculate and return the h1 heuristic for a given state
        # The h1 heuristic is the number of tiles out of place from their goal position

        # Flatten the lists for comparison
        state_flat_list = sum(state, [])
        goal_flat_list = sum(self.goal_state, [])
        heuristic = 0

        # Iterate through the lists and compare elements
        for i, j in zip(state_flat_list, goal_flat_list):
            if i != j:
                heuristic += 1

        
        return heuristic

    def calculate_h2_heuristic(self, state):
        # Calculates and return the h2 heuristic for a given state
        # The h2 hueristic for the eight puzzle is defined as the sum of the Manhattan distances of all the tiles
        # Manhattan distance is the sum of the absolute value of the x and y difference of the current tile position from its goal state position

        state_dict = {}
        goal_dict = {}
        heuristic = 0
        
        # Create dictionaries of the current state and goal state
        for row_index, row in enumerate(state):
            for col_index, element in enumerate(row):
                state_dict[element] = (row_index, col_index)
        
        for row_index, row in enumerate(self.goal_state):
            for col_index, element in enumerate(row):
                goal_dict[element] = (row_index, col_index)
                
        for tile, position in state_dict.items():
            # Do not count the distance of the blank 
            if tile == 0:
                pass
            else:
                # Calculate heuristic as the Manhattan distance
                goal_position = goal_dict[tile]
                heuristic += (abs(position[0] - goal_position[0]) + abs(position[1] - goal_position[1]))

        return heuristic

    def calculate_total_cost(self, node_depth, state, heuristic):
        # Returns the total cost of a state given its depth and the heuristic
        # Total cost in a-star is path cost plus heuristic. The path cost in this case is depth, or the number of moves from the start state to the current state because all moves have the same cost

        if heuristic == "h2":
            return node_depth + self.calculate_h2_heuristic(state)
        elif heuristic == "h1":
            return node_depth + self.calculate_h1_heuristic(state)
    
    def a_star(self, heuristic="h2", max_nodes=10000, print_solution=True):
        # Performs a-star search
        # Prints the list of solution moves and the solution length

        # Need a dictionary for the frontier and for the expanded nodes
        frontier_nodes = {}
        expanded_nodes = {}
        
        self.starting_state = copy.deepcopy(self.state)
        current_state = copy.deepcopy(self.state)
        # Node index is used for indexing the dictionaries and to keep track of the number of nodes expanded
        node_index = 0

        # Set the first element in both dictionaries to the starting state
        # This is the only node that will be in both dictionaries
        expanded_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.calculate_total_cost(0, current_state, heuristic), "depth": 0}
        
        frontier_nodes[node_index] = {"state": current_state, "parent": "root", "action": "start",
                                   "total_cost": self.calculate_total_cost(0, current_state, heuristic), "depth": 0}
        

        failure = False

        # all_nodes keeps track of all nodes on the frontier and is the priority queue. Each element in the list is a tuple consisting of node index and total cost of the node. This will be sorted by the total cost and serve as the priority queue.
        all_frontier_nodes = [(0, frontier_nodes[0]["total_cost"])]

        # Stop when maximum nodes have been considered
        while not failure:

            # Get current depth of state for use in total cost calculation
            current_depth = 0
            for node_num, node in expanded_nodes.items():
                if node["state"] == current_state:
                    current_depth = node["depth"]

            # Find available actions corresponding to current state
            available_actions, _, _ = self.get_available_actions(current_state)

            # Iterate through possible actions 
            for action in available_actions:
                repeat = False

                # If max nodes reached, break out of loop
                if node_index >= max_nodes:
                    failure = True
                    print("No Solution Found in first {} nodes generated".format(max_nodes))
                    self.num_nodes_generated = max_nodes
                    break

                # Find the new state corresponding to the action and calculate total cost
                new_state = self.move(current_state, action)
                new_state_parent = copy.deepcopy(current_state)

                # Check to see if new state has already been expanded
                for expanded_node in expanded_nodes.values():
                    if expanded_node["state"] == new_state:
                        if expanded_node["parent"] == new_state_parent:
                            repeat = True

                # Check to see if new state and parent is on the frontier
                # The same state can be added twice to the frontier if the parent state is different
                for frontier_node in frontier_nodes.values():
                    if frontier_node["state"] == new_state:
                        if frontier_node["parent"] == new_state_parent:
                            repeat = True

                # If new state has already been expanded or is on the frontier, continue with next action     
                if repeat:
                    continue

                else:
                    # Each action represents another node generated
                    node_index += 1
                    depth = current_depth + 1

                    # Total cost is path length (number of steps from starting state) + heuristic
                    new_state_cost = self.calculate_total_cost(depth, new_state, heuristic)

                    # Add the node index and total cost to the all_nodes list
                    all_frontier_nodes.append((node_index, new_state_cost))

                    # Add the node to the frontier 
                    frontier_nodes[node_index] = {"state": new_state, "parent": new_state_parent, "action": action, "total_cost": new_state_cost, "depth": current_depth + 1}

            # Sort all the nodes on the frontier by total cost
            all_frontier_nodes = sorted(all_frontier_nodes, key=lambda x: x[1])

            # If the number of nodes generated does not exceed max nodes, find the best node and set the current state to that state
            if not failure:
                # The best node will be at the front of the queue
                # After selecting the node for expansion, remove it from the queue
                best_node = all_frontier_nodes.pop(0)
                best_node_index = best_node[0]
                best_node_state = frontier_nodes[best_node_index]["state"]
                current_state = best_node_state

                # Move the node from the frontier to the expanded nodes
                expanded_nodes[best_node_index] = (frontier_nodes.pop(best_node_index))
                
                # Check if current state is goal state
                if self.goal_check(best_node_state):
                    # Create attributes for the expanded nodes and the frontier nodes
                    self.expanded_nodes = expanded_nodes
                    self.frontier_nodes = frontier_nodes
                    self.num_nodes_generated = node_index + 1

                    # Display the solution path
                    self.success(expanded_nodes, node_index, print_solution)
                    break 
                    
    def local_beam(self, k=1, max_nodes = 10000, print_solution=True):
        # Performs local beam search to solve the eight puzzle
        # k is the number of successor states to consider on each iteration
        # The evaluation function is h1 + h2, at each iteration, the next set of nodes will be the k nodes with the lowest score

        self.starting_state = copy.deepcopy(self.state)
        starting_state = copy.deepcopy(self.state)
        # Check to see if the current state is already the goal
        if starting_state == self.goal_state:
            self.success(node_dict={}, num_nodes_generated=0)

        # Create a reference dictionary of all states generated
        all_nodes= {}

        # Index for all nodes dictionary
        node_index = 0

        all_nodes[node_index] = {"state": starting_state, "parent": "root", "action": "start"}

        # Score for starting state
        starting_score = self.calculate_h1_heuristic(starting_state) + self.calculate_h2_heuristic(starting_state)

        # Available nodes is all the possible states that can be accessed from the current state stored as an (index, score) tuple
        available_nodes = [(node_index, starting_score)]
                
        failure = False
        success = False

        while not failure:

            # Check to see if the number of nodes generated exceeds max nodes
            if node_index >= max_nodes:
                failure = True
                print("No Solution Found in first {} generated nodes".format(max_nodes))
                break
              
            # Successor nodes are all the nodes that can be reached from all of the available states. At each iteration, this is reset to an empty list  
            successor_nodes = []

            # Iterate through all the possible nodes that can be visited
            for node in available_nodes:

                repeat = False

                # Find the current state
                current_state = all_nodes[node[0]]["state"]

                # Find the actions corresponding to the state
                available_actions, _, _ = self.get_available_actions(current_state)

                # Iterate through each action that is allowed
                for action in available_actions:
                    # Find the successor state for each action
                    successor_state = self.move(current_state, action)

                    # Check if the state has already been seen
                    for node_num, node in all_nodes.items():
                        if node["state"] == successor_state:
                            if node["parent"] == current_state:
                                repeat = True

                    # Check if the state is the goal state
                    # If the best state is the goal, stop iteration
                    if successor_state == self.goal_state:	
                        all_nodes[node_index] = {"state": successor_state, 
                                "parent": current_state, "action": action}
                        self.expanded_nodes = all_nodes
                        self.num_nodes_generated = node_index + 1
                        self.success(all_nodes, node_index, print_solution)
                        success = True
                        break

                    if not repeat:
                        node_index += 1
                        # Calculate the score of the state
                        score = (self.calculate_h1_heuristic(successor_state) + self.calculate_h2_heuristic(successor_state))
                        # Add the state to the list of of nodes
                        all_nodes[node_index] = {"state": successor_state, "parent": current_state, "action": action}
                        # Add the state to the successor_nodes list
                        successor_nodes.append((node_index, score))
                    else:
                        continue

                    
            # The available nodes are now all the successor nodes sorted by score
            available_nodes = sorted(successor_nodes, key=lambda x: x[1])

            # Choose only the k best successor states
            if k < len(available_nodes):
                available_nodes = available_nodes[:k]
            if success == True:
            	break  
                
    def success(self, node_dict, num_nodes_generated, print_solution=True):
        # Once the solution has been found, prints the solution path and the length of the solution path
        if len(node_dict) >= 1:

            # Find the final node
            for node_num, node in node_dict.items():
                if node["state"] == self.goal_state:
                    final_node = node_dict[node_num]
                    break

            # Generate the solution path from the final node to the start node
            solution_path = self.generate_solution_path(final_node, node_dict, path=[([[0, 1, 2], [3, 4, 5], [6, 7, 8]], "goal")])
            solution_length = len(solution_path) - 2

        else:
            solution_path = []
            solution_length = 0
        
        self.solution_path = solution_path 

        if print_solution:
            # Display the length of solution and solution path
            print("Solution found!")
            print("Solution Length: ", solution_length)

            # The solution path goes from final to start node. To display sequence of actions, reverse the solution path
            print("Solution Path", list(map(lambda x: x[1], solution_path[::-1])))
            print("Total nodes generated:", num_nodes_generated)
        
    def generate_solution_path(self, node, node_dict, path):
        # Return the solution path for display from final (goal) state to starting state
        # If the node is the root, return the path
        if node["parent"] == "root":
            # If root is found, add the node and then return
            path.append((node["state"], node["action"]))
            return path

        else:
            # If the node is not the root, add the state and action to the solution path
            state = node["state"]
            parent_state = node["parent"]
            action = node["action"]
            path.append((state, action))

            # Find the parent of the node and recurse
            for node_num, expanded_node in node_dict.items():
                if expanded_node["state"] == parent_state:
                    return self.generate_solution_path(expanded_node, node_dict, path)