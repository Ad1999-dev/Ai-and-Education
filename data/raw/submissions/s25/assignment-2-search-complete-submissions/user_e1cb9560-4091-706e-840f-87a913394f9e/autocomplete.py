from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self, name = None, is_word = False):
        self.children = {}
        self.name = name
        self.is_word = is_word
        self.frequency = 0

class frontierNode(): # This class is to represent node as it is added to the frontier, while keeping track of weight and string on the way
    def __init__(self, node: Node, weight: float, string: str):
        self.data = (node, weight, string)

    def __lt__(self, other):
        return self.data[1] < other.data[1]

    def __le__(self, other):
        return self.data[1] <= other.data[1]

    def __gt__(self, other):
        return self.data[1] > other.data[1]

    def __ge__(self, other):
        return self.data[1] >= other.data[1]

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children: # if char is not in the children, add it, then keep traversing
                    node.children[char] = Node(name = char)
                node = node.children[char]
                node.frequency += 1 # this count how many this the node has been visited. i.e. the frequency of that letter after some prefix
            node.is_word = True

    # Helpers
    def traversePrefix(self, prefix) -> Node: # traverse from root until the end of the given prefix
        node = self.root
        for char in prefix:
            node = node.children[char]
        return node

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        output = []
        queue = deque()
        node = self.traversePrefix(prefix)
        queue.append(frontierNode(node, 0, prefix))
        while (len(queue) > 0):
            nodeToProcess = queue.popleft()
            if (nodeToProcess.data[0].is_word):
                output.append(nodeToProcess.data[2])
            for child in nodeToProcess.data[0].children.values():
                queue.append(frontierNode(child, nodeToProcess.data[1] + 1/child.frequency, nodeToProcess.data[2] + child.name))
        return output
    
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        output = []
        stack = deque()
        node = self.traversePrefix(prefix)
        stack.append(frontierNode(node, 0, prefix))
        while (len(stack) > 0):
            nodeToProcess = stack.pop()
            if (nodeToProcess.data[0].is_word):
                output.append(nodeToProcess.data[2])
            for child in nodeToProcess.data[0].children.values():
                stack.append(frontierNode(child, nodeToProcess.data[1] + 1/child.frequency, nodeToProcess.data[2] + child.name))
        return output

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        output = []
        pq = []
        node = self.traversePrefix(prefix)
        heapq.heappush(pq, frontierNode(node, 0, prefix))
        while (len(pq) > 0):
            nodeToProcess = heapq.heappop(pq)
            if (nodeToProcess.data[0].is_word):
                output.append(nodeToProcess.data[2])
            for child in nodeToProcess.data[0].children.values():
                heapq.heappush(pq, frontierNode(child, nodeToProcess.data[1] + 1/child.frequency, nodeToProcess.data[2] + child.name))
        return output
