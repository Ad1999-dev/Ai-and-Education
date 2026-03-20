from collections import deque, OrderedDict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.cost = 0
    
    def __eq__(self, other):
        return (self.children == other.children) and (self.cost == other.cost)

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return (self.cost < other.cost)

    def __gt__(self, other):
        return (self.cost > other.cost)

    def __le__(self, other):
        return (self < other) or (self == other)

    def __ge__(self, other):
        return (self > other) or (self == other)


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    

    def build_tree(self, document):
        def traverse(node):
            node.children = OrderedDict(sorted(node.children.items()))
            if len(node.children) > 0:
                for char, child in node.children.items():
                    child.cost = node.cost/child.cost
                    traverse(child)

        for word in document.split():
            node = self.root
            node.cost += 1

            for char in word:
                #TODO for students
                if not node.children.__contains__(char):
                    node.children[char] = Node()
                node = node.children.get(char)
                node.cost += 1
                
            if not node.children.__contains__("EOS"):
                node.children["EOS"] = Node()

            node = node.children.get("EOS")
            node.cost += 1

        traverse(self.root)
        

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    #TODO for students!!!
    def suggest_bfs(self, prefix):
        if len(prefix) < 1:
            return []
        
        node = self.root

        for char in prefix:
            if not node.children.__contains__(char):
                return [prefix]
            
            node = node.children.get(char)

        frontier = deque([(node, prefix)])
        suggestions = []

        while len(frontier) > 0: 
            node, pre = frontier.popleft()
            if "EOS" in node.children:
                suggestions.append(pre)

            for char, child in node.children.items():
                if char != "EOS":
                    frontier.append((child, pre + char))

        return suggestions if len(suggestions) > 0 else [prefix]
        
    
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        if len(prefix) < 1:
            return []

        node = self.root
        for char in prefix:
            if not node.children.__contains__(char):
                return [prefix]
            node = node.children.get(char)

        frontier = deque([(node, prefix)])
        suggestions = []

        while len(frontier) > 0: 
            node, pre = frontier.pop()

            if "EOS" in node.children:
                suggestions.append(pre)
            
            for char, child in node.children.items():
                if char != "EOS":
                    frontier.append((child, pre + char))
        
        return suggestions if len(suggestions) > 0 else [prefix]


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        if len(prefix) < 1:
            return []

        frontier = []
        heapq.heapify(frontier)

        node = self.root
        for char in prefix:
            if not node.children.__contains__(char):
                return [prefix]
            node = node.children.get(char)

        heapq.heappush(frontier, (node.cost, node, prefix))  
        suggestions = []
        
        while len(frontier) > 0:  
            cost, node, pre = heapq.heappop(frontier)

            if "EOS" in node.children:
                suggestions.append(pre)

            for char, child in node.children.items():
                if char != "EOS":
                    heapq.heappush(frontier, (cost + child.cost, child, pre + char))

        return suggestions if len(suggestions) > 0 else [prefix]
