from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isWordEnd = False
        self.frequency = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for i, character in enumerate(word):
                if character not in node.children:
                    node.children[character] = Node()
                node = node.children[character]

                if i + 1 < len(word):
                    next_character = word[i + 1]
                    if next_character not in node.frequency:
                        node.frequency[next_character] = 0
                    node.frequency[next_character] += 1
            
            node.isWordEnd = True 

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for character in prefix:
            if character in node.children:
                node = node.children[character]
            else:
                return []  

        suggestions = []
        queue = deque([(node, prefix)])

        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.isWordEnd:  
                suggestions.append(current_prefix)

            for character, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + character))

        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for character in prefix:
            if character in node.children:
                node = node.children[character]
            else:
                return []  

        suggestions = []
        stack = [(node, prefix)]  # LIFO

        while stack:
            current_node, current_prefix = stack.pop()  
            if current_node.isWordEnd:  
                suggestions.append(current_prefix)

            for character, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + character))

        return suggestions



    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for character in prefix:
            if character in node.children:
                node = node.children[character]
            else:
                return []  

        suggestions = []
        priority_queue = []
        heapq.heappush(priority_queue, (0, prefix, node)) 

        while priority_queue:
            cost, current_prefix, current_node = heapq.heappop(priority_queue)

            if current_node.isWordEnd:  
                suggestions.append(current_prefix)

            for character, child_node in current_node.children.items():
                path_cost = self.calc_path_cost(current_prefix, character)
                heapq.heappush(priority_queue, (cost + path_cost, current_prefix + character, child_node))

        return suggestions

    def calc_path_cost(self, prefix, character):
        node = self.root
        for c in prefix:
            if c in node.children:
                node = node.children[c]
            else:
                return float('inf')
        
        if character in node.frequency:
            frequency2 = node.frequency[character]
            return 1 / frequency2 if frequency2 > 0 else float('inf')

        return float('inf')
