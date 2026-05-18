from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        # add end of word tracker
        self.end_word = False


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        # storing path costs for the tree
        self.path_cost = {}
    
    
    def build_tree(self, document):

        for word in document.split():

            node = self.root
            prefix = ""

            for char in word:
                #TODO for students

                # get new prefix and add to prefix list if new
                prefix = prefix + char
                if (prefix in self.path_cost):
                    self.path_cost[prefix] += 1
                else:
                    self.path_cost[prefix] = 1

                # check if char is already a child node
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.end_word = True

        # inverse the path costs for the prefix
        for pref in self.path_cost:
            self.path_cost[pref] = 1/self.path_cost[pref]


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):

        node = self.root
         # find last letter of prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            # if prefix is not in the tree, return empty result
            else:
                return []

        # intialize list of suggested words
        result = []
        
        # initialize a queue and add starting node
        q = deque()
        q.append((node, prefix))
        
        while (q):
        
            # popleft to get next node
            curr_node, curr_prefix = q.popleft()
            # check if current node is end of word
                # if end of word, add to result
            if curr_node.end_word:
                result.append(curr_prefix)
            # add all children into queue
            for char, child_node in curr_node.children.items():
                new_prefix = curr_prefix + char
                q.append((child_node, new_prefix))
            
        return result

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):

        node = self.root
        # get to last node of prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            # if prefix is not in the tree, return empty result
            else:
                return []
            
        # intialize list of suggested words
        result = []

        # initialize a queue and add starting node
        q = deque()
        q.append((node, prefix))

        # queue to put all nodes in
        while(q):
            
            # get last node in queue
            curr_node, curr_prefix = q.pop()
            if curr_node.end_word:
                result.append(curr_prefix)

            # put all of the children into the queue
            for char, child_node in curr_node.children.items():
                new_prefix = curr_prefix + char
                q.append((child_node, new_prefix))
            
        return result
            


    #TODO for students!!!
    def suggest_ucs(self, prefix):

        node = self.root
        # get to last node of prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            # if prefix is not in the tree, return empty result
            else:
                return []
            
        # intialize list of suggested words
        result = []

        # initialize heap queue to store costs with nodes and prefix
        q = []
        heapq.heappush(q, (self.path_cost[prefix], prefix, node))

        while (q):

            # pop element with smallest cost
            min_cost, min_prefix, min_node = heapq.heappop(q)
    
            # check if its the end of a word
            if min_node.end_word:
                result.append(min_prefix)
            
            # add children to the heap queue
            for char, child_node in min_node.children.items():
                new_prefix = min_prefix + char
                new_cost = self.path_cost[new_prefix]
                heapq.heappush(q, (new_cost, new_prefix, child_node))

        return result

