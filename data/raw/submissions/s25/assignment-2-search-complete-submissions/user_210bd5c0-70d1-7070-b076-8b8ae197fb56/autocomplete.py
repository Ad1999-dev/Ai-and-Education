from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.parent = None
        self.letter = ""
        self.cost = 0.0
        # self.is_word = False
    
    def __lt__(self, other):
        return self.cost < other.cost

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs

    
    
    def build_tree(self, document):
        node = self.root
        node.children = {}
        for word in document.split():
            node = self.root
            for char in word:
                if node.children.get(char) == None:
                    node.children[char] = Node()
                    node.children[char].parent = node
                    node.children[char].letter = char
                node = node.children[char]
            node.children["end"] = Node()
            node.children["end"].parent = node
        self.cost()
        print(str(self.root.cost))
            
    def cost(self, node = None):
        if node == None:
            node = self.root
        for letter, child in node.children.items():
            if letter == "end":
                node.cost += 1
            else:
                node.cost += 1/self.cost(child)
        node.cost = 1/node.cost
        #print("node: "+node.letter+", cost: "+ str(node.cost))
        return node.cost


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    
    def suggest_bfs(self, prefix):
        node = find_prefix(self, prefix)
        if node == None:
            return []
        suggestions = []
        frontier = []
        frontier.extend(node.children.items())
        while len(frontier) != 0:
            letter, child = next(iter(frontier))

            if letter == "end":
                word = ""
                cur_node = child
                while cur_node.parent != None:
                    word += cur_node.letter
                    word += "+"
                    cur_node = cur_node.parent
                letters = word.split("+")
                letters.reverse()
                word = "".join(letters)
                suggestions.append(word)
                frontier.pop(0)
            else:
                frontier.extend(child.children.items())
                frontier.pop(0)
        #print(suggestions)
        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = find_prefix(self, prefix)
        if node == None: 
            return []
        suggestions = []
        for letter, child in node.children.items():
            suggestions.extend(self.helper(child, prefix+letter))
       # print(suggestions)
        return suggestions
    
    def helper(self, node, prefix):
        if len(node.children) == 0:
            return [prefix.replace("end", "")]
        result = []
        for letter, child in node.children.items():
            result.extend(self.helper(child, prefix+letter))
        return result


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = find_prefix(self, prefix)
        if node == None:
            return []
        suggestions = []
        frontier = []
        frontier.extend(node.children.items())
        heapq.heapify(frontier)
        while len(frontier) != 0:
            letter, child = heapq.heappop(frontier)

            if letter == "end":
                word = ""
                cur_node = child
                while cur_node.parent != None:
                    word += cur_node.letter
                    word += "+"
                    cur_node = cur_node.parent
                letters = word.split("+")
                letters.reverse()
                word = "".join(letters)
                suggestions.append(word)
            else:
                for item in child.children.items():
                    heapq.heappush(frontier, item)
        #print(suggestions)
        return suggestions

        

def find_prefix(self, prefix):
        node = self.root
        for char in prefix:
            if node.children.get(char) == None:
                return None
            node = node.children[char]
        return node