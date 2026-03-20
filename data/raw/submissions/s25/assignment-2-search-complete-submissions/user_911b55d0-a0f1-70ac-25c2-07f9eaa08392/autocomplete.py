from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.cnt = 0
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if (char not in node.children):
                    node.children[char] = Node()
                node = node.children[char]
                node.cnt += 1
            node.children[''] = Node()
            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children):
                return []
            node = node.children[char]
        queue = deque([(node, prefix)])
        words = []
        while (queue):
            if ('' in queue[0][0].children):
                words.append(queue[0][1])
                queue.append((queue[0][0].children[''], queue[0][1]))
            for char in queue[0][0].children:
                queue.append((queue[0][0].children[char], queue[0][1] + char))
            queue.popleft()
        return words

        

    

    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children):
                return []
            node = node.children[char]
        queue = deque([(node, prefix)])
        visited = set()
        words = []
        while (queue):
            current_node = queue[0][0]
            current_prefix = queue[0][1]
            if(current_node not in visited):
                if ('' in current_node.children):
                    words.append(current_prefix)
                for char in current_node.children:
                    queue.appendleft((current_node.children[char], current_prefix + char))
                visited.add(current_node)
            else:
                queue.popleft()
        return words


    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if (char not in node.children):
                return []
            node = node.children[char]
        words = []
        minCost = [(0, prefix, node)]
        while (minCost):
            tuple = heapq.heappop(minCost)
            pathCost = tuple[0]
            current_node = tuple[2]
            current_prefix = tuple[1]
            if ('' in current_node.children):
                words.append(current_prefix)
            for char in current_node.children:
                if(current_node.children[char].cnt > 0):
                    heapq.heappush(minCost, (1/current_node.children[char].cnt + pathCost, current_prefix + char, current_node.children[char]))
        return words