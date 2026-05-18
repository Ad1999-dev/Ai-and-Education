from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self, char):
        self.char = char  
        self.children = {}  
        self.is_end_of_word = False  
        self.frequency = 0
        self.transition_frequency = {}

class Autocomplete():
    def __init__(self, parent=None):
        self.root = Node("")
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        for word in document.split():
            current_node = self.root
            for char in word:
                if char not in current_node.children:
                    current_node.children[char] = Node(char)
                    current_node.transition_frequency[char] = 1 
                else:
                    current_node.transition_frequency[char] += 1
                current_node = current_node.children[char] 
            current_node.is_end_of_word = True
            current_node.frequency += 1

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        suggestions = []
        queue = deque()  

        current_node = self.root
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return suggestions 

        queue.append((current_node, prefix))

        while queue:
            node, word = queue.popleft()

            if node.is_end_of_word:
                suggestions.append(word)

            for child in node.children.values():
                queue.append((child, word + child.char))

        return suggestions

    def suggest_dfs(self, prefix):
        suggestions = []
        stack = []  

        current_node = self.root
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return suggestions  

        stack.append((current_node, prefix))

        while stack:
            node, word = stack.pop()

            if node.is_end_of_word:
                suggestions.append(word)

            for child in reversed(list(node.children.values())):
                stack.append((child, word + child.char))

        return suggestions

    def suggest_ucs(self, prefix):
        suggestions = []
        pq = []  

        current_node = self.root
        current_cost = 0
        for char in prefix:
            if char in current_node.children:
                current_cost += 1 / current_node.transition_frequency[char]
                current_node = current_node.children[char]
            else:
                return suggestions 

        heapq.heappush(pq, (current_cost, prefix, current_node))

        while pq:
            path_cost, word, node = heapq.heappop(pq)

            if node.is_end_of_word:
                suggestions.append(word)

            for child_char, child_node in node.children.items():
                transition_cost = 1 / node.transition_frequency[child_char]
                heapq.heappush(pq, (path_cost + transition_cost, word + child_char, child_node))

        return suggestions
