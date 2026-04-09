from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.curr_word = ""
        self.children = {}
        self.frequency = {}
        self.is_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            curr_pre = ""
            for char in word:
                curr_pre += char
                if char not in node.children:
                    node.children[char] = Node()
                if char not in node.frequency:
                    node.frequency[char] = 0
                node.frequency[char] += 1
                node = node.children[char]
                node.curr_word = curr_pre
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]

        queue = [node]
        complete_words = []

        while queue:
            curr_node = queue.pop(0)
           
            if curr_node.is_word:
                complete_words.append(curr_node.curr_word)
            
            for element in curr_node.children.values():
                queue.append(element)
                 
        return complete_words

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]

        stack = [node]
        complete_words = []

        while stack:
            curr_node = stack.pop(0)
           
            if curr_node.is_word:
                complete_words.append(curr_node.curr_word)

            for element in curr_node.children.values():
                stack.insert(0, element)  

        return complete_words


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            node = node.children[char]

        pq = []
        complete_words = []

        heapq.heappush(pq, (0, id(node), node))

        while pq:
            initial_cost, id_node, curr_node = heapq.heappop(pq)   

            if curr_node.is_word:
                complete_words.append(curr_node.curr_word)

            for char, element in curr_node.children.items():
                freq_count = curr_node.frequency[char]
                additional_cost = 1 / freq_count
                heapq.heappush(pq, (initial_cost + additional_cost, id(element), element))

        return complete_words
