from collections import deque
import heapq
import random
import string

class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequencies = {}
        self.path_costs = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs
        self.build_tree(document)

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.frequencies:
                    node.frequencies[char] = 0
                node.frequencies[char] += 1
                node.path_costs[char] = 1.0 / node.frequencies[char]
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def get_prefix_node(self, prefix): # helper function to get start node at end of prefix
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def suggest_bfs(self, prefix):
        start_node = self.get_prefix_node(prefix)
        if not start_node: # no autocomplete suggestions available
            return []
        suggestions = []
        queue = deque([(start_node, prefix)]) # double-ended queue for bfs
        while queue:
            curr_node, curr_word = queue.popleft()
            if curr_node.is_word: # reached last char of a word in autocomplete tree, add to suggestions
                suggestions.append(curr_word)
            for char in sorted(curr_node.children.keys()): # enqueue child nodes for bfs
                queue.append((curr_node.children[char], curr_word + char))
        return suggestions

    def suggest_dfs(self, prefix):
        start_node = self.get_prefix_node(prefix)
        if not start_node: # no autocomplete suggestions available
            return []
        suggestions = []
        stack = [(start_node, prefix)] # stack for dfs
        while stack:
            curr_node, curr_word = stack.pop()
            if curr_node.is_word: # reached last char of a word in autocomplete tree, add to suggestions
                suggestions.append(curr_word)
            for char in sorted(curr_node.children.keys(), reverse=True): # stack child nodes for dfs
                stack.append((curr_node.children[char], curr_word + char))
        return suggestions

    def suggest_ucs(self, prefix):
        start_node = self.get_prefix_node(prefix)
        if not start_node: # no autocomplete suggestions available
            return []
        suggestions = []
        pq = [(0, prefix, start_node)] # priority queue for ucs
        visited = set() # keep track of visited nodes
        while pq:
            cost, word, curr_node = heapq.heappop(pq) # pop node with lowest path cost
            if word in visited: # skip if already visited
                continue
            visited.add(word)
            if curr_node.is_word: # reached last char of a word in autocomplete tree, add to suggestions
                suggestions.append(word)
            for char, child_node in curr_node.children.items(): # push child nodes to priority queue with their path costs
                if word + char not in visited:
                    path_cost = cost + curr_node.path_costs[char]
                    heapq.heappush(pq, (path_cost, word + char, child_node))
        return suggestions