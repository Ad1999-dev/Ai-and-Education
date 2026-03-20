from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children = {}
        self.word = False
        self.period = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            #enumerate: (index, character)
            for char in enumerate(word):
                if char[1] in node.children:
                    if char[0] == len(word) - 1:
                        node.children[char[1]].word = True
                        node.period = 1 / ((1 / node.period) + 1)
                    node = node.children[char[1]]
                else:
                    node.children[char[1]] = Node()
                    node.period = 1
                    if char[0] == len(word) - 1:
                        node.children[char[1]].word = True
                        node.period = 1 / ((1 / node.period) + 1)
                    node = node.children[char[1]]

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):
        result = []
        tracker = prefix
        node = self.root

        #make sure the prefix is in the tree
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
                continue
            else:
                return []

        #Check if the prefix itself is a word
        if node.word:
            result.append(tracker)

        def check(node, temp):
            if node.word:
                result.append(temp)
            return

        queue = deque([node.children, tracker])

        while queue:
            current = queue.popleft()
            temp = queue.popleft()
            for key in current.keys():
                check(current[key], temp + key)
                queue.append(current[key].children)
                queue.append(temp + key)

        return result

    def suggest_dfs(self, prefix):
        #assigning variables
        result = []
        temp = prefix
        node = self.root
        visited = set()
        
        #make sure the prefix is in the tree
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
                continue
            else:
                return []
    
        #recursive function
        def dfs_recursion(node, temp, visited):

            #exploring discovered/not visited nodes
            if temp not in visited:
                if node.word:
                    result.append(temp)
                visited.add(temp)

                #check the children
                for k in node.children.keys():
                    dfs_recursion(node.children[k], temp + k, visited)

        #call recursive function with node after prefix checking
        dfs_recursion(node, temp, visited)
        
        #return the list
        return result

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        result = []
        temp = prefix
        node = self.root
        visited = set()

        #make sure the prefix is in the tree
        for char in prefix:
            if char in node.children.keys():
                node = node.children[char]
                continue
            else:
                return []

        #Check if the prefix itself is a word
        if node.word:
            result.append(temp)

        #function to resort the dictionary by smallest period
        def smallest_period(diction):
            new_keys = {i: diction[i].period for i in diction.keys()}
            return dict(sorted(new_keys.items(), key=lambda item: item[1]))

        #recursive function
        def ucs_recursion(node, temp, visited):

            #exploring discovered/not visited nodes
            if temp not in visited:
                if node.word:
                    result.append(temp)
                visited.add(temp)

                #check the children
                for k in smallest_period(node.children):
                    ucs_recursion(node.children[k], temp + k, visited)

        #call recursive function with node after prefix checking
        ucs_recursion(node, temp, visited)

        #return the list
        return result
