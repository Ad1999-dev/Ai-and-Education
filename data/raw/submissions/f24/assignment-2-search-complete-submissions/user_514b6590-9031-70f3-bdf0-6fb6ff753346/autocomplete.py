from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isFinal = False
        self.freq = 1
        self.cost = 1

    def __lt__(self, other):
        return (self.cost < other.cost)

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for index, char in enumerate(word):
                #TODO for students
                char = char.lower()
                if char in node.children:
                    node = node.children[char]
                    node.freq = node.freq + 1
                    node.cost = 1 / node.freq
                    if index == len(word) - 1:
                        node.isFinal = True
                    continue
                
                node.children[char] = Node()
                node = node.children[char]
                node.cost = 1 / node.freq
                if index == len(word) - 1:
                    node.isFinal = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        #Get to end of prefix in tree
        for char in prefix:
            if char in node.children:
                node = node.children[char]
                continue
            else:
                return []
        #BFS search and return
        queue = deque([prefix])
        nodes = deque([node])
        final = []
        while queue:
            current = queue.popleft()
            node = nodes.popleft()
            if node.isFinal:
                final.append(current)
            keys = list(node.children.keys())
            words = []
            for key in keys:
                words.append(current + key)
            queue.extend(words)
            nodes.extend(list(node.children.values()))
        
        return final

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        #Get to end of prefix in tree
        for char in prefix:
            if char in node.children:
                node = node.children[char]
                continue
            else:
                return []
        stack = [prefix]
        nodes = [node]
        final = []
        while stack:
            current = stack.pop()
            node = nodes.pop()
            if node.isFinal:
                final.append(current)
            words = []
            for key in list(node.children.keys()):
                words.append(current + key)
            stack.extend(reversed(words))
            nodes.extend(reversed(list(node.children.values())))
        
        return final

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        #Get to end of prefix in tree
        for char in prefix:
            if char in node.children:
                node = node.children[char]
                continue
            else:
                return []
        pQueue = []
        nodes = []
        final = []
        heapq.heappush(pQueue, (node.cost, prefix))
        heapq.heappush(nodes, (node.cost, node))
        explored = set()

        #Traversal of pQueue
        while pQueue:
            cost, current = heapq.heappop(pQueue)
            cost2, node = heapq.heappop(nodes)

            if node.isFinal:
                final.append(current)
            
            explored.add(node)

            keys = list(node.children.keys())
            values = list(node.children.values())
            for i in range(len(keys)):
                if values[i] not in explored:
                    heapq.heappush(pQueue, (values[i].cost + cost, current + keys[i]))
                    heapq.heappush(nodes, (values[i].cost + cost, values[i]))
                
        
        return final