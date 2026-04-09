from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

        if document:
            self.build_tree(document)
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True  


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    
    def _find_node(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
    
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []

        queue = deque([(node, prefix)])
        results = []

        while queue:
            current_node, path = queue.popleft()
            if current_node.is_word:
                results.append(path)
            for char, child in sorted(current_node.children.items()):
                queue.append((child, path + char))

        return results

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self._find_node(prefix)
        if not node:
            return []

        stack = [(node, prefix)]
        results = []

        while stack:
            current_node, path = stack.pop()
            if current_node.is_word:
                results.append(path)
            for char, child in sorted(current_node.children.items(), reverse=True):
                stack.append((child, path + char))

        return results


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        result = []
        node = self._find_node(prefix)
        if not node:
            return []

        pq = []
        heapq.heappush(pq, (0, prefix, node))
        
        while pq:
            cost, word, current_node = heapq.heappop(pq)
            if current_node.is_word:
                result.append(word)
            for char, child in current_node.children.items():
                heapq.heappush(pq, (cost + 1, word + char, child))
        return result   
