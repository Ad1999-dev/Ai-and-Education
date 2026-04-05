from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.priority = 1

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                char = char.lower()
                if char not in node.children:
                    node.children[char] = Node()
                else:
                    node.children[char].priority = ((node.children[char].priority ** -1) + 1) ** -1
                node = node.children[char]
            node.children['$'] = Node() # Using $ to denote end of word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # First we traverse the tree following the characters of prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]
            else: # Tree does not contain any words with this prefix, return empty list
                return []
        
        # Now we do BFS with nodes containing $ children as a goal node
        word_list = []
        
        # frontier will contain a 2-tuple with the node and the word that node represents
        frontier = deque()
        frontier.append((curr, prefix))
        while len(frontier) > 0:
            curr, word = frontier.popleft()
            for char, node in curr.children.items():
                if char == '$':
                    word_list.append(word)
                else:
                    frontier.append((node, word + char))
        return word_list

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # First we traverse the tree following the characters of prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]
            else: # Tree does not contain any words with this prefix, return empty list
                return []
        
        # Now we do DFS with nodes containing $ children as a goal node
        word_list = []
        
        # frontier will contain a 2-tuple with the node and the word that node represents
        frontier = deque()
        frontier.append((curr, prefix))
        while len(frontier) > 0:
            curr, word = frontier.pop()
            for char, node in curr.children.items():
                if char == '$':
                    word_list.append(word)
                else:
                    frontier.append((node, word + char))
        return word_list


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # First we traverse the tree following the characters of prefix
        curr = self.root
        for char in prefix:
            if char in curr.children:
                curr = curr.children[char]
            else: # Tree does not contain any words with this prefix, return empty list
                return []
        
        # Now we do UCS with nodes containing $ children as a goal node
        word_list = []
        
        # frontier will contain a 3-tuple with the priority, the word that node represents, and the node
        frontier = []
        heapq.heappush(frontier, (curr.priority, prefix, curr))
        while len(frontier) > 0:
            prior_priority, word, curr = heapq.heappop(frontier)
            for char, node in curr.children.items():
                if char == '$':
                    word_list.append(word)
                else:
                    heapq.heappush(frontier, (prior_priority + node.priority, word + char, node))
        return word_list
