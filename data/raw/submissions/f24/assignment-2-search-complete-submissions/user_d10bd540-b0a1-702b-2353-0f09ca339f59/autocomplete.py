from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.end_of_word = False
        self.frequency = 0
        self.path_costs = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children: 
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1
            node.end_of_word = True
        
        for word in document.split():
            node = self.root
            for char in word:
                total_frequency = sum(children.frequency for children in node.children.values())
                for child_char, child_node in node.children.items():
                    node.path_costs[child_char] = total_frequency / child_node.frequency
                node = node.children[char]

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]  
        queue = deque([(node, prefix)])
        suggestions = []
        while queue:
            current_node, current_prefix = queue.popleft()
            if current_node.end_of_word:
                suggestions.append(current_prefix)
            for char, char_node in current_node.children.items():
                queue.append((char_node, current_prefix + char))  
        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]  
        stack = [(node, prefix)]
        suggestions = []
        while stack:
            current_node, current_prefix = stack.pop()
            if current_node.end_of_word:
                suggestions.append(current_prefix)
            for char, char_node in current_node.children.items():
                stack.append((char_node, current_prefix + char))
        return suggestions

    # TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]         
        pq = []
        suggestions = []
        heapq.heappush(pq, (0, id(node), node, prefix))  
        while pq:
            current_cost, _, current_node, current_word = heapq.heappop(pq)
            if current_node.end_of_word:
                suggestions.append((current_word))
            for char, child_node in current_node.children.items():
                path_cost = current_node.path_costs[char]
                cumulative_cost = current_cost + path_cost
                heapq.heappush(pq, (cumulative_cost, id(child_node), child_node, current_word + char)) 
        return suggestions

# testing part
with open("genZ.txt", "r") as file:
    document = file.read()

autocomplete = Autocomplete()
autocomplete.build_tree(document)

# Test with the prefix 'th'
prefix = "li"
suggestions_bfs = autocomplete.suggest_bfs(prefix)
print("BFS Suggestions: ", suggestions_bfs)

suggestions_dfs = autocomplete.suggest_dfs(prefix)
print("DFS Suggestions: ", suggestions_dfs)

suggestions_ucs = autocomplete.suggest_ucs(prefix)
print("UCS Suggestions: ", suggestions_ucs)
