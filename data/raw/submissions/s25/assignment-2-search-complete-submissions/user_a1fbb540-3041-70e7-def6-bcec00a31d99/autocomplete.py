from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, letter=None):
        self.children = {}
        self.letter = letter
        self.cost = 0 # this will be defined with build_tree
        self.is_word = False

    def __gt__(self, other):
        return self.cost > other.cost
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def __geq__(self, other):
        return self.cost == other.cost

    def calc_cost(self, letter=''):
        # Base case: If no children, return 0 instances
        if not self.children:
            return 0 

        instances = 0  # Initialize instances count

        # Only start counting instances from the first level of children
        for child in self.children.values():
            # If this child node's letter matches the given letter, increment instances
            if child.letter == letter:
                instances -= 1

            # Recurse into child nodes to count instances of 'letter'
            instances += child.calc_cost(letter=letter)
        
        return instances

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    

    def build_tree(self, document):
        nodes = set()

        for word in document.split():
            node = self.root

            for char in word[:-1]:
                if char not in node.children: #If character is not in the tree, create a new node for the char
                    node.children[char] = Node(char) # initialize new child node
                    nodes.add(node.children[char])
                
                node = node.children[char] # iterate to new child node

            char = word[-1]
            if char not in node.children: #If character is not in the tree, create a new node for the char
                node.children[char] = Node(char) # initialize new child node
                nodes.add(node.children[char])
                
            node = node.children[char] # iterate to new child node
            node.is_word = True

        for node in nodes:
            node.cost = node.calc_cost(node.letter)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        startnode = self.root
        suggestions = []

        # Navigate down to the end of the prefix in the trie
        for char in prefix:
            if char in startnode.children:
                startnode = startnode.children[char]
            else:
                # No such prefix exists, return empty suggestions
                return suggestions

        # BFS to find all words with given prefix
        queue = deque([(startnode, prefix)])  # Queue contains tuple of (node, current_word)

        while queue:
            node, current_word = queue.popleft()
        
            # Check if it's a terminal node (end of a valid word)
            if node.is_word:
                suggestions.append(current_word)
        
            # Enqueue all children
            for char, child_node in node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        startnode = self.root
        suggestions = []

        # Navigate down to the end of the prefix in the trie
        for char in prefix:
            if char in startnode.children:
                startnode = startnode.children[char]
            else:
                # No such prefix exists, return empty suggestions
                return suggestions

        # BFS to find all words with given prefix
        queue = deque([(startnode, prefix)])  # Queue contains tuple of (node, current_word)

        while queue:
            node, current_word = queue.pop()
        
            # Check if it's a terminal node (end of a valid word)
            if node.is_word:
                suggestions.append(current_word)
        
            # Enqueue all children
            for char, child_node in node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        startnode = self.root
        suggestions = []

        # Navigate to the end of the prefix node in the trie
        for char in prefix:
            if char in startnode.children:
                startnode = startnode.children[char]
            else:
                return suggestions  # No such prefix exists

        # Use a priority queue (min-heap) for UCS
        priority_queue = [(0, startnode, prefix)]  # (cumulative cost, node, current_word)

        while priority_queue:
            current_cost, node, current_word = heapq.heappop(priority_queue)

            if node.is_word:
                suggestions.append((current_word))  # accumulated cost

            # Enqueue all children with updated cumulative cost
            for char, child_node in node.children.items():
                new_cost = current_cost + child_node.cost
                heapq.heappush(priority_queue, (new_cost, child_node, current_word + char))

        return suggestions
