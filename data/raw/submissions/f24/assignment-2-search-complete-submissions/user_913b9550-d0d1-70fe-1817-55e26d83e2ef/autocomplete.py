from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
    def __lt__(self,other):
        return self

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            flag = False
            for char in word:
                for key in node.children:
                    if key == char:
                        node.children[key][1] += 1
                        node = node.children[key][0]
                        flag = True
                        break
                if flag:
                    flag = False
                else:
                    node.children[char]=[Node(), 1]
                    node = node.children[char][0]
                    #print("new node = ", char)
            
            node.children["word"] = word
        """queue = [self.root]
        while len(queue) > 0:
            node = queue.pop(0)
            print(node.children)
            for child in node.children:
                if type(child) != str:
                    queue.append(child)"""





    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        #traverse the prefix
        node = self.root
        flag = False
        for char in prefix:
            for key in node.children:
                
                if key == char:
                    #print(key)
                    #print (node.children[key][0].children)
                    node = node.children[key][0]
                    flag = True
                    break
            if not flag:
                print("Error: prefix not found in tree")
                break
        #BFS
        queue = [node]
        suggestions = []
        while len(queue) > 0:
            node = queue.pop(0)
            for child in node.children:
                if child == "word":
                    suggestions.append(node.children["word"])
                else:
                    queue.append(node.children[child][0])
        return suggestions



    def recurse_dfs(self, node):
        suggestions = []
        for child in node.children:
            if child == "word":
                suggestions.append(node.children["word"])
            else:
                for item in self.recurse_dfs(node.children[child][0]):
                    suggestions.append(item)
        return suggestions



    #TODO for students!!!
    def suggest_dfs(self, prefix):
        #traverse the prefix
        node = self.root
        flag = False
        for char in prefix:
            for key in node.children:
                
                if key == char:
                    #print(key)
                    #print (node.children[key][0].children)
                    node = node.children[key][0]
                    flag = True
                    break
            if not flag:
                print("Error: prefix not found in tree")
                break
        #DFS
        return self.recurse_dfs(node)

    


    def sortbykey(self, item: list [Node, int]):
        return item[1]

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        #traverse the prefix
        node = self.root
        flag = False
        for char in prefix:
            for key in node.children:
                if key == char:
                    node = node.children[key][0]
                    flag = True
                    break
            if not flag:
                print("Error: prefix not found in tree")
                break
        #UCS
        queue = []
        heapq.heappush(queue, (1, node))
        ans = []
        while True:
            
            print(queue)
            try:
                currnode = heapq.heappop(queue)
            except IndexError:
                break
            for child in currnode[1].children:
                if child == "word":
                    print(currnode[1].children["word"])
                    ans.append(currnode[1].children["word"])
                else:
                    val = 1/currnode[1].children[child][1]
                    nod = currnode[1].children[child][0]
                    heapq.heappush(queue, (val,nod))
        return ans




