from collections import deque, defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.char = ''
        self.ending = False
        self.parent = None
        self.path_cost = float('inf')  # Initialize path cost to infinity


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.char_frequency = defaultdict(int)
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    new_node = Node()
                    new_node.char = char
                    new_node.parent = node
                    node.children[char] = new_node
                node = node.children[char]  # Move to the child node for the next character
                self.char_frequency[char] += 1
            node.ending = True
        #return self

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        suggestions = []

        # Find the node corresponding to the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions  # No words with that prefix

        # Perform BFS to find all words starting from this node
        queue = deque([node])
        while queue:
            current_node = queue.popleft()
            # If the current node marks the end of a word, add it to suggestions
            if current_node.ending:
                suggestions.append(''.join(self._collect_chars(current_node)))

            # Queue up the children for further exploration
            for child in current_node.children.values():
                queue.append(child)

        return suggestions

    def _collect_chars(self, node):
        # Backtrack to collect characters to form the word
        chars = []
        while node.parent is not None:
            chars.append(node.char)
            node = node.parent
        return reversed(chars)  # Return reversed to get the word in the correct order


        

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return[]
        
            # This will hold the suggestions
        suggestions = []
        
        # Call the recursive helper to collect suggestions
        self._dfs_helper(node, prefix[:len(prefix)-1], suggestions)
        
        return suggestions
    
    def _dfs_helper(self, node, current_word, suggestions):
        # Collect the word if we are at a valid node not equal to root
        # Visit each child node
        for child in node.children.values():
            self._dfs_helper(child, current_word + node.char, suggestions)
        if node.ending:
            word = current_word + node.char
            suggestions.append(word)


        



    #TODO for students!!!
    def count_frequencies(self, node):
        # Count the number of words that are complete from this node
        if not node:
            return 0
            
        total_count = 0
        
        # If this node marks the end of a word, it counts as 1
        if node.ending:
            total_count += 1
        
        # Traverse child nodes
        for char, child in node.children.items():
            total_count += self.count_frequencies(child)
            
            # Store the frequency of words that can be formed with the character `char`
            self.char_frequency[char] = total_count
        
        return total_count

    def ucs(self, node, prefix, suggestions):
        if node.ending:
            suggestions.append(prefix)
        
        for char, child in node.children.items():
            self.ucs(child, prefix + char, suggestions)

    def suggest_ucs(self, prefix):
        # Find the node corresponding to the prefix
        node = self.root
        suggestions = []
        
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # No suggestions if the prefix is not found
        
        # Perform DFS to find all completions
        self.ucs(node, prefix, suggestions)
        return suggestions