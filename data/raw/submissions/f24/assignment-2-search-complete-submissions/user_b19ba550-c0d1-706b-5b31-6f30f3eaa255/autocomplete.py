from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.end = False
        self.frequency_counter = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    # Need to build a trie given the words from the document
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                # create a node for each letter in the word, and have it be on a simple path
                # if there is a path which exists already for the word, then continue down that path until new nodes need to be created

                if char not in node.children:
                    node.children[char] = Node()
                if char not in node.frequency_counter:
                    node.frequency_counter[char] = 0
                node.frequency_counter[char] += 1
                node = node.children[char] 
            node.end = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for letter in prefix:
            if letter in node.children:
                node = node.children[letter]
            else:
                return []
        
        autocomplete = []
        queue = [(node, prefix)] 
        
        while queue:
            cur_node, cur_word = queue.pop(0)  

            if cur_node.end:
                autocomplete.append(cur_word)

            for char, child_node in cur_node.children.items():
                queue.append((child_node, cur_word + char))  

        return autocomplete

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for letter in prefix:
            if letter in node.children:
                node = node.children[letter]
            else:
                return []
    
        autocomplete = []
        stack = [(node, prefix)] 

        while stack:
            cur_node, cur_word = stack.pop()  

            if cur_node.end:
                autocomplete.append(cur_word)

            for char, child_node in cur_node.children.items():
                stack.append((child_node, cur_word + char))  

        return autocomplete


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for letter in prefix:
            if letter in node.children:
                node = node.children[letter]
            else:
                return []  # If prefix path doesn't exist, return empty list

        autocomplete = []
        p_queue = [(0, prefix, node)]  

        while p_queue:
            cost, cur_word, cur_node = heapq.heappop(p_queue)

            # Check if we are at a complete word
            if cur_node.end:
                autocomplete.append(cur_word)

            for char, child_node in cur_node.children.items():
                new_cost = cost + (1 / cur_node.frequency_counter[char])
                heapq.heappush(p_queue, (new_cost, cur_word + char, child_node))

        return autocomplete
