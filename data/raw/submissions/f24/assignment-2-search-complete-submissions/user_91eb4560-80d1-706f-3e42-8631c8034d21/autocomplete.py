from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs
 #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                if char in node.frequency:
                    node.frequency[char] += 1
                else:
                    node.frequency[char] = 1
                node = node.children[char]

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        curr = self.root
        for letter in prefix:
            if letter in curr.children:
                curr = curr.children[letter]
            else:
                return [] 
        queue = deque([(curr, prefix)])
        sug = []
        while queue:
            node, word = queue.popleft()
            if not node.children:
                sug.append(word)
            else:
                for next_char, child_node in node.children.items():
                    queue.append((child_node, word + next_char))
        return sug

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        curr = self.root
        for letter in prefix:
            if letter in curr.children:
                curr = curr.children[letter]
            else:
                return [] 
        to_visit = [(curr, prefix)]
        sug = []
        while to_visit:
            node, word = to_visit.pop()
            if not node.children: 
                sug.append(word)
            else:
                for next_letter, next_node in node.children.items():
                    to_visit.append((next_node, word + next_letter))
        return sug 

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        curr = self.root
        for letter in prefix:
            if letter not in curr.children:
                return []
            curr = curr.children[letter]
        pq, sug = [(0, prefix, curr)], []
        while pq:
            cost, word, node = heapq.heappop(pq)
            if not node.children:
                sug.append(word)
            else:
                for next_char, child_node in node.children.items():
                    new_cost = cost + 1 / node.frequency[next_char]
                    heapq.heappush(pq, (new_cost, word + next_char, child_node))
        return sug
