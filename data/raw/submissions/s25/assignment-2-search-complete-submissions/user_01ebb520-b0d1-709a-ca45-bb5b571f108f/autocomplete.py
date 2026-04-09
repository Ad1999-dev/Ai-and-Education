from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = ''
        self.costs = {}
        self.count = 0

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


    def build_tree(self, document):
        for word in document.split():
            node = self.root
            node.count += 1

            for char in word:
                # checks if next letter in word not represented in tree
                if char not in node.children:
                    # if not, adds char to dictionary as a key, value is a new node
                    node.children[char] = Node()
                    
                    # initializes self.word, self.cost[char] for new node
                    node.children[char].word = node.word + char
                    node.costs[node.word + char] = 0

                # updates count for child node
                node.children[char].count += 1
                
                # updates value of node
                node = node.children[char]
            # sets is_word to true for current node
            node.is_word = True
        
        # updates self.costs for each node using bfs
        curr = self.root
        frontier = deque([curr])

        while frontier:
            curr = frontier[0]
            frontier.popleft()

            # update self.costs for curr
            for char, child in curr.children.items():
                frontier.append(child)
                curr.costs[child.word] = curr.count / curr.children[char].count

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    # function that finds node cooresponding to prefix
    def find_prefix(self, prefix):
        node = self.root
        for char in prefix:
            # if prefix not in tree, stop and print message
            if char not in node.children.keys():
                print('No words cooresponding to given prefix.')
                exit(1)
            node = node.children[char]
        return node
   

    def suggest_bfs(self, prefix):
        node = self.find_prefix(prefix)

        frontier = deque([node])

        # compeletes bfs traversal
        while frontier: 
            #choose 
            node = frontier[0]

            # remove node from frontier
            frontier.popleft()

            # prints node if it's a word
            if node.is_word:
                print(node.word)

            # expands
            for n in node.children.values():
                frontier.append(n)


    def suggest_dfs(self, prefix):
        node = self.find_prefix(prefix)

        frontier = deque([node])

        # completes stack-based dfs traversal
        while frontier:
            # choose
            node = frontier[-1]

            # remove node from frontier
            frontier.pop()

            # prints node if it's a word
            if node.is_word:
                print(node.word)

            # expands
            for n in node.children.values():
                frontier.append(n)


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.find_prefix(prefix)

        heap = [(1, node.word, node)]

        while heap:
            currCost, currWord, curr = heapq.heappop(heap)

            # if curr is a word, print it
            if curr.is_word:
                print(curr.word)

            # explore children nodes
            for char, child in curr.children.items():
                totalCost = currCost * curr.costs[currWord + char]
                heapq.heappush(heap, (totalCost, child.word, child))
