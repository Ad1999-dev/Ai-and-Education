from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.value = None
        self.pred = None
        self.children = {}
        self.inverse_cost = 0
        self.end_of_word = False
        # self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children.keys():
                    node.children[char] = Node()
                    node.children[char].value = char
                    node.children[char].pred = node
                node = node.children[char]
            node.end_of_word = True
            # calculate the cost
            while node.pred != self.root:
                node.inverse_cost += 1
                node = node.pred

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        if not prefix:
            return []
        
        # traverse the tree to find the node correspond to the end of the prefix in the tree
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else: # prefix doesn't match any words in the txt file
                return []

        # run bfs
        frontier = []
        suggested = []
        frontier.append(node)
        while frontier:
            node = frontier.pop(0)
            # adding word into suggested if we've found the entire word
            if node.end_of_word:
                word_node = node
                path = []
                while word_node != self.root:
                    path.insert(0, word_node.value)
                    word_node = word_node.pred
                suggested.append(''.join(path))
            for child in node.children.values():
                frontier.append(child)
        return suggested
            
    def suggest_dfs(self, prefix):
        if not prefix:
            return []
        
        # traverse the tree to find the node correspond to the end of the prefix in the tree
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else: # prefix doesn't match any words in the txt file
                return []

        # run dfs
        frontier = []
        suggested = []
        frontier.append(node)
        while frontier:
            node = frontier.pop(-1)
            # adding word into suggested if we've found the entire word
            if node.end_of_word:
                word_node = node
                path = []
                while word_node != self.root:
                    path.insert(0, word_node.value)
                    word_node = word_node.pred
                suggested.append(''.join(path))
            for child in node.children.values():
                frontier.append(child)
        return suggested

    def suggest_ucs(self, prefix):
        if not prefix:
            return []
        
        # traverse the tree to find the node correspond to the end of the prefix in the tree
        node = self.root
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
            else: # prefix doesn't match any words in the txt file
                return []
        
        frontier = [] # holds elements of (node (Node), total_cost (double))
        suggested = []
        for child in node.children.values():
            frontier.append((child, 1/child.inverse_cost))
        while frontier: 
            # determine the node with the smallest cost to remove
            min_tuple = frontier[0] 
            min_cost = min_tuple[1]
            for t in frontier:
                if t[1] < min_cost:
                    min_tuple = t
                    min_cost = min_tuple[1]
            frontier.remove(min_tuple)
            # backtrack to add word to suggested if it is the end of the word
            if min_tuple[0].end_of_word:
                word_node = min_tuple[0]
                path = []
                while word_node != self.root:
                    path.insert(0, word_node.value)
                    word_node = word_node.pred
                suggested.append(''.join(path))
            # add children to the queue
            for child in min_tuple[0].children.values():
                frontier.append((child, (1/child.inverse_cost)+min_tuple[1]))
        return suggested
