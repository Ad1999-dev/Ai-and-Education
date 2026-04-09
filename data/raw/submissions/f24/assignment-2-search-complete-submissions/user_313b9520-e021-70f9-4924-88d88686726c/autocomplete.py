from collections import deque
from collections import defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.char_frequency = defaultdict(int)

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                node.char_frequency[char] +=1
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end = True
                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        queue = deque([(node, prefix)])
        result = []

        while queue:
            curr_node, curr_word = queue.popleft()

            if curr_node.is_end:
                result.append(curr_word)

            for char, child in curr_node.children.items():
                queue.append((child, curr_word + char))


        return result

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        stack = deque([(node, prefix)])
        result = []

        while stack:
            curr_node, curr_word = stack.pop()

            if curr_node.is_end:
                result.append(curr_word)

            for char, child in curr_node.children.items():
                stack.append((child, curr_word + char))


        return result




    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        queue = []
        result = []
        heapq.heappush(queue, (0, prefix, node))

        while queue:
            cost, curr_word, curr_node = heapq.heappop(queue)

            if curr_node.is_end:
                result.append(curr_word)
            
            for char, child in curr_node.children.items():
                count = curr_node.char_frequency[char]
                path_cost = cost + (1/count)
                heapq.heappush(queue, (path_cost, curr_word + char, child))
        return result
