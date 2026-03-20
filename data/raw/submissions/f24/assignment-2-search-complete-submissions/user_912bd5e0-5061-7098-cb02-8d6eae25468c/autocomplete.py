from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.suffix = False
        self.cnt = 0
    
    def __lt__(self, other):
        return self.cnt < other.cnt

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.cnt += 1
            node.suffix = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    #TODO for students!!!

    def suggest_bfs(self, prefix=""):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        q =[]
        suggestions=[]
        for child in node.children:
            q.append([node.children[child], prefix+child])
        while len(q) != 0:
            if len(q[0][0].children) == 0:
                suggestions.append(q[0][1])
            else:
                if q[0][0].suffix:
                    suggestions.append(q[0][1])
                for child in q[0][0].children:
                    q.append([q[0][0].children[child], q[0][1]+child])
            q.pop(0)
        print(suggestions)
        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        def dfs_traversal(node, curprefix):
            if len(node.children) == 0:
                suggestions.append(curprefix)
                return
            if node.suffix:
                suggestions.append(curprefix)
            for child in node.children:
                dfs_traversal(node.children[child],curprefix+child)
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        s =[]
        suggestions=[]
        dfs_traversal(node,prefix)
        print(suggestions)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        h = []
        suggestions = []
        heapq.heappush(h,(0,node,prefix))
        while len(h) != 0:
            count, node, prefix = h[0]
            if len(node.children) == 0:
                suggestions.append(prefix)
            else:
                if node.suffix:
                    suggestions.append(prefix)
                for child in node.children:
                    print(node.children , suggestions)
                    if (node.cnt > 0):
                        heapq.heappush(h, (count+(1/(node.cnt)),node.children[child],prefix+child))
            heapq.heappop(h)
        return suggestions
