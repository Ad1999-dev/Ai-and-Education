from collections import deque, defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = defaultdict(int)  # Stores frequency of each child character

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        

    
    
    def build_tree(self, document):
        """Builds a trie and records character sequence frequencies at each step."""
        for word in document.split():
            node = self.root
            for i, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = Node()
                
                # Update frequency of character in the current node
                node.freq[char] += 1
                node = node.children[char]
            
            node.is_word = True  # Mark end of a word

    
    
                


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]  # Move deeper in the tree
            else:
                return []  # No matches found, return empty list

        queue = deque([(node, prefix)])  # Start BFS from this node
        suggestions = []

        while queue:
            current_node, word = queue.popleft()  # Dequeue first element

            if current_node.is_word:  # Now we can properly check for full words
                suggestions.append(word)

            for char, child in current_node.children.items():
                queue.append((child, word + char))  # Enqueue child nodes

        return suggestions

    

    #TODO for students!!!
    def dfs_recursive(self, node, prefix, suggestions):
        if node.is_word:
            suggestions.append(prefix)
        
        for char, child in node.children.items():
            self.dfs_recursive(child, prefix + char, suggestions)

    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        suggestions = []
        self.dfs_recursive(node, prefix, suggestions)
        return suggestions
        
    """def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # Prefix not found

        suggestions = []
        stack = [(node, prefix)]  # Stack stores (current node, current word)

        while stack:
            curr_node, word = stack.pop()
            if curr_node.is_word:
                suggestions.append(word)
            
            # Push children onto stack (reverse order to maintain proper DFS traversal)
            for char, child in sorted(curr_node.children.items(), reverse=True):
                stack.append((child, word + char))

        return suggestions"""

    def suggest_ucs(self, prefix):
        """Uses UCS to find words with the lowest accumulated path cost."""
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]  # Move deeper in the tree
            else:
                return []  # No matches found, return empty list

        # Min-heap for UCS: (accumulated_cost, word, node)
        pq = []
        heapq.heappush(pq, (0, prefix, node))  # Start with cost 0 and the prefix
        suggestions = []

        while pq:
            accumulated_cost, word, current_node = heapq.heappop(pq)
            
            if current_node.is_word:
                suggestions.append(word)  # Add the word to suggestions if it's a complete word
            
            # Traverse all children, prioritizing by accumulated path cost
            for char, child in current_node.children.items():
                # Calculate the cost for this character (inverse frequency)
                char_cost = 1 / current_node.freq[char]
                # Push the child node onto the heap with updated cost and word
                heapq.heappush(pq, (accumulated_cost + char_cost, word + char, child))
        
        return suggestions