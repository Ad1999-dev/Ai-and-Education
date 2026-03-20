from collections import deque
import heapq
import random
import string


class Node:
    #TODO: added the letter attribute
    def __init__(self, letter = None):
        self.letter = letter
        self.children = {}
        self.final = False
        self.freq = 0
    
    # def __lt__(self, other):
    #     # return True
    #     return 1 / self.freq < 1 / other.freq

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        # self.suggest = self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                node = node.children.setdefault(char, Node(char))
                node.freq += 1 # store frequencies, calculate inverses later
            node.final = True
        # print(self.root.children)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO: helper function to get to the node of the last letter

    def __traverse_prefix(self, pref): # private - returns the node of the last letter
        node = self.root
        for char in pref:
            if char not in node.children:
                return None
            else:
                node = node.children[char]
        
        return node
        
    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestions = []
        prefix_end = self.__traverse_prefix(prefix)
        if prefix_end == None:
            return suggestions
        
        q = deque([(prefix_end, prefix)])
        
        while q:
            cur, s = q.popleft()
            if cur.final:
                suggestions.append(s)
            for nextChar in cur.children:
                q.append((cur.children[nextChar], s + nextChar))

        return suggestions


        # pass

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        # search for last letter of prefix node
        prefix_end = self.__traverse_prefix(prefix)
        if prefix_end == None:
            return []
        
        def dfs(cur, suffix = ""):
            if cur.final: # end of word
                suggestions.append(prefix + suffix)
            for nextChar in cur.children:
                dfs(cur.children[nextChar], suffix + nextChar) # could use join for faster runtime
        
        dfs(prefix_end, "")
        
        return suggestions
                
        # now node = last letter in prefix, we have to dfs it

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        prefix_end = self.__traverse_prefix(prefix)
        if prefix_end == None:
            return []
        
        i = 0
        
        pq = [(0, i, prefix_end, prefix)] # first 0 is for the path cost, second 0 is to break ties (order in which it goes in the queue)

        heapq.heapify(pq)

        while pq:
            path, order, node, word = heapq.heappop(pq)
            if node.final:
                suggestions.append(word)
            for k in node.children:
                i += 1
                heapq.heappush(pq, (path + 1 / node.children[k].freq, i, node.children[k], word + node.children[k].letter))
        
        return suggestions
            