from collections import deque, defaultdict
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.cost = 0  # Default cost

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs
    
    def build_tree(self, document):
        frequency = defaultdict(int)
        total_count = defaultdict(int)

        # Calculate frequencies of each character following a prefix
        for word in document.split():
            for i in range(len(word)):
                prefix = word[:i]
                char = word[i]
                frequency[(prefix, char)] += 1
                total_count[prefix] += 1

        # Build the tree and calculate path costs
        for word in document.split():
            node = self.root
            for i, char in enumerate(word):
                prefix = word[:i]
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            
                # Calculate path cost as 1 / frequency
                node.cost = 1 / frequency[(prefix, char)]
        
            # Marking the end of the word with '~' and setting its cost to 0
            node.children['~'] = Node()
            node.children['~'].cost = 0

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        # Traverse the tree to the node corresponding to the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        queue = deque([(node, prefix)])
        suggestions = []

        # Perform BFS
        while queue:
            current_node, current_word = queue.popleft()

            # If this node has the '~' symbol, it's the end of a word
            if '~' in current_node.children:
                suggestions.append(current_word)

            # Add all children to the queue, constructing words as we go (ignoring '~')
            for char, child_node in current_node.children.items():
                if char != '~':  # Skip the end-of-word marker
                    queue.append((child_node, current_word + char))
        return suggestions

    def suggest_dfs(self, prefix):
        suggestions = []
        node = self.root
        # Traverse to the node corresponding to the last character of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return suggestions
    
        # Depth-First Search to find all words from the current node
        stack = deque([(node, prefix)])
        while stack:
            current_node, current_prefix = stack.pop()
            if '~' in current_node.children:  # End of a word
                suggestions.append(current_prefix)
            for char in sorted(current_node.children.keys(), reverse=True):  # Reverse to simulate stack behavior
                stack.append((current_node.children[char], current_prefix + char))
        return suggestions

    def suggest_ucs(self, prefix):
        # Traverse to the node corresponding to the last character of the prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        # Perform UCS
        queue = deque([(node, prefix, 0)])
        suggestions = []
        visited = set()
        
        while queue:
            # Sorting the queue by cost
            queue = deque(sorted(queue, key=lambda x: x[2]))
            
            # Pop the node with the lowest cost
            current_node, current_word, current_cost = queue.popleft()
            if current_word in visited:
                continue
            visited.add(current_word)

            if '~' in current_node.children:
                suggestions.append(current_word)
            
            # Adding all children nodes to the queue with their cumulative cost
            for char, child_node in current_node.children.items():
                if char != '~':
                    queue.append((child_node, current_word + char, current_cost + child_node.cost))
        
        return suggestions