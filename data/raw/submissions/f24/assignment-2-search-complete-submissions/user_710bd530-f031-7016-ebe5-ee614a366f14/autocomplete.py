from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.visits = 0 # number of times the node has been accessed in build_tree (no of words passing through or ending at this node)
        self.end = 0 # number of words ending at the node

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        # self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.suggest = self.suggest_ucs

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            i = 0
            for char in word:
                #TODO for students
                if char not in node.children:
                    charNode = Node()
                    node.children[char] = charNode
                node.visits += 1
                # print(node.visits)
                node = node.children[char]
                if i == len(word) - 1:
                    node.end += 1
                i += 1

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    # Helper method to find the node corresponding to the last letter of the prefix. Used in suggest_bfs, suggest_dfs, and suggest_ucs
    def find_start(self, start, prefix):
        for ch in prefix:
            if ch not in start.children:
                # invalid prefix
                return None
            start = start.children[ch]
        return start

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        result = []
        start = self.find_start(self.root, prefix)
        if not start:
            return result
        queue = deque()
        # every element of the queue is [word, next node]
        for ch, node in start.children.items():
            queue.append([prefix + ch, node])
        while len(queue) != 0:
            word, node = queue.popleft()
            if not node or node.end > 0:
                result.append(word)
            for ch, nxt in node.children.items():
                queue.append([word + ch, nxt])
        return result

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        result = []
        start = self.find_start(self.root, prefix)
        if not start:
            return result
        stack = []
        # every element of the stack is [word, next node]
        successors = list(start.children.items())[::-1]
        for ch, node in successors:
            stack.append([prefix + ch, node])
        while len(stack) != 0:
            word, node = stack.pop()
            if not node or node.end > 0:
                result.append(word)
            successors = list(node.children.items())[::-1]
            for ch, nxt in successors:
                stack.append([word + ch, nxt])
        return result

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        result = []
        start = self.find_start(self.root, prefix)
        if not start:
            return result
        pq = []
        heapq.heapify(pq) # min-heap
        # every element of pq is (cost, word, next node)
        for ch, node in start.children.items():
            cnt = node.visits + node.end
            cost = 1 / cnt
            heapq.heappush(pq, (cost, prefix + ch, node))
        while len(pq) != 0:
            cost, word, nxt = heapq.heappop(pq)
            if not nxt or nxt.end > 0:
                result.append(word)
            for ch, node in nxt.children.items():
                cnt = node.visits + node.end
                nxtCost = 1 / ((1 / cost) + cnt)
                heapq.heappush(pq, (nxtCost, word + ch, node))
        return result
    