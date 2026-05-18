from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if (char not in node.children):
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end_of_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char in node.children):
                node = node.children[char]
            else:
                return []

        queue = deque([(node, prefix)])
        results = []

        while (queue):
            current_node, current_prefix = queue.popleft()

            if (current_node.is_end_of_word):
                results.append(current_prefix)

            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char))
        
        return results

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char in node.children):
                node = node.children[char]
            else:
                return []

        stack = [(node, prefix)]
        results = []

        while stack:
            current_node, current_prefix = stack.pop()

            if (current_node.is_end_of_word):
                results.append(current_prefix)

            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))

        return results


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if (char in node.children):
                node = node.children[char]
            else:
                return []

        pq = [(0, node, prefix)]
        results = []

        while (pq):
            cost, current_node, current_prefix = heapq.heappop(pq)

            if (current_node.is_end_of_word):
                results.append(current_prefix)

            for char, child_node in current_node.children.items():
                new_cost = cost + 1
                heapq.heappush(pq, (new_cost, child_node, current_prefix + char))

