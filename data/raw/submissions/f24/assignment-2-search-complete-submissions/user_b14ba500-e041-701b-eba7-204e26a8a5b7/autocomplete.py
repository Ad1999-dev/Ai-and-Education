from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.freq = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            print(f"building for word: {word}")
            for char in word:
                char = char.lower() # make all
                if char in node.children.keys():
                    node = node.children[char]
                    node.freq += 1
                else:
                    node.children[char] = Node()
                    node = node.children[char]
                    node.freq += 1
            node.is_end_of_word = True # mark last node as the end of a word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else:
                return [] # no words in the tree which have the prefix
        frontier = deque([(node, prefix)])
        words = []
        while len(frontier) > 0:
            node, pref = frontier.popleft()
            if node.is_end_of_word:
                words.append(pref)
            for char in node.children.keys():
                child = node.children[char]
                frontier.append((child, pref + char))
        return words


    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else:
                return []  # no words in the tree which have the prefix
        frontier = deque([(node, prefix)])
        words = []
        while len(frontier) > 0:
            node, pref = frontier.pop()
            if node.is_end_of_word:
                words.append(pref)
            for char in node.children.keys():
                child = node.children[char]
                frontier.append((child, pref + char))
        return words


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        words = []
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else:
                return []  # no words in the tree which have the prefix
        frontier = [(0, 1, node, prefix)]
        heapq.heapify(frontier)
        count = 1
        while len(frontier) > 0:
            cost, _, node, pref = heapq.heappop(frontier)
            if node.is_end_of_word:
                words.append(pref)
            for char in node.children.keys():
                count += 1
                child = node.children[char]
                child_cost = cost + 1/child.freq
                heapq.heappush(frontier, (child_cost, count, child, pref + char))
        return words