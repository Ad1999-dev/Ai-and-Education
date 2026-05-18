from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = ''
        self.times_visited = 0

# Helper class for the UCS search
class Path:
    def __init__(self, node, cost=0):
        self.cost_to_last_node = cost
        self.last_node = node

    def __lt__(self, other_path):
        return self.cost_to_last_node < other_path.cost_to_last_node

    def __gt__(self, other_path):
        return self.cost_to_last_node > other_path.cost_to_last_node

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                # Check if the char is present at this level
                if char not in node.children:
                    # If its not, add it to the node, and create a new node for the next character
                    new_node = Node()
                    node.children[char] = new_node
                    new_node.word = node.word + char # Update the word made so far to make the searching easier

                node = node.children[char] # Then progress to the next node for the next character
                node.times_visited += 1 # We need to update the cost

            # We are now at the end of the word, so make that node an end node
            node.is_end = True


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!

    # HELPER FUNCTION FOR METHDOS
    def skip_prefix(self, prefix):
        # Find the node where the prefix ends
        cur_node = self.root
        for char in prefix:
            if char not in cur_node.children:
                return None # The prefix does not exist in the tree
            
            cur_node = cur_node.children[char]

        return cur_node


    def suggest_bfs(self, prefix):
        start_node = self.skip_prefix(prefix) # Find the node up to what the user has already typed
        if start_node == None:
            return [] # The prefix did not exist
    
        possible_words = []
        frontier_queue = deque([start_node])

        while len(frontier_queue) != 0:
            search_node = frontier_queue.popleft()
            if search_node.is_end: # This node represents the end of a word
                possible_words.append(search_node.word)

            # Now we need to add its children to the queue
            frontier_queue.extend(search_node.children.values())

        return possible_words

            


    #TODO for students!!!
    def suggest_dfs(self, prefix):
        start_node = self.skip_prefix(prefix)
        if start_node == None:
            return [] # The prefix did not exist
        
        possible_words = []
        frontier_stack = [start_node]

        while len(frontier_stack) != 0:
            search_node = frontier_stack.pop()
            if search_node.is_end:
                possible_words.append(search_node.word)

            frontier_stack.extend(search_node.children.values())

        return possible_words


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        start_node = self.skip_prefix(prefix)
        if start_node == None:
            return [] # The prefix did not exist
        
        possible_words = []
        frontier_pqueue = [Path(start_node, 0)]

        while len(frontier_pqueue) != 0:
            path = heapq.heappop(frontier_pqueue)
            if path.last_node.is_end:
                possible_words.append(path.last_node.word)

            # Get the next possible nodes from this one
            child_nodes = path.last_node.children.values()

            # Convert them to paths and calculate the total cost to go there
            for node in child_nodes:
                total_cost_to_node = path.cost_to_last_node + (1 / node.times_visited) # We are adding the cost of the next edge to the cost of the path so far
                heapq.heappush(frontier_pqueue, Path(node, total_cost_to_node))

        return possible_words