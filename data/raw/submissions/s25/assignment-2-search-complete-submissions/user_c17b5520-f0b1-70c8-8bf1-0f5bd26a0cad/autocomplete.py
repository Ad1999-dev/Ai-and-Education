from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.frequency = 0
        self.wordCount = 0
        
    def __lt__(self, other):
        return self.wordCount < other.wordCount
        
        

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                ## check if root's child has the first char
                if(char not in node.children):
                    ##then add it 
                    node.children[char] = Node()
                
                
                node = node.children[char]
                node.wordCount += 1 

            node.is_word = True



    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
   
        ## non recursive method
        node = self.root
        ans = []
   
        ## first we want to check if the prefix exists
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        ##now we know the prefix exists

        ##creating the queue 
        queue = [(node, prefix)]

        while queue:
            cur_node,word = queue.pop(0)

            if(cur_node.is_word):
                    ans.append(word)

            for child in cur_node.children:
                queue.append((cur_node.children[child], word + child))
               
        return ans
    
      # node = self.root
        # ans = []

        # for char in prefix:
        #     if char not in node.children:
        #         return []
        #     node = node.children[char]
        
        # queue = [(node, prefix)]

        # def helper(n, p, queue):
        #     store_n, store_p = queue.pop(0)
        #     if(n.is_word()):
        #         ans.append(n)
        #     for child in n.children:
        #         queue.add(n)


        # return helper(node, prefix, queue)
    
            
            
        

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        ans = []

        for char in prefix:
            if(char not in node.children):
                return []
            node = node.children[char]
        
        stack = [(node, prefix)]

        while stack:
            cur_node, word = stack.pop()
            if(cur_node.is_word):
                ans.append(word)
            ##add all of its neighbors
            # for child in sorted(cur_node.children.keys(), reverse=True):
            for child in cur_node.children:
                stack.append((cur_node.children[child], word + child))

        return ans



        


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        ans = []

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        heap = []
        heapq.heappush(heap, (0, node, prefix))

        while heap:
            count, n, word = heapq.heappop(heap)   

            if(n.is_word):
                ans.append(word)

            for child in n.children:
                heapq.heappush(heap, ((float(1) / n.children[child].wordCount) + count, n.children[child], word + child))
            
        return ans
            

                                                          


# store = Autocomplete()


# with open('genZ.txt', 'r') as file:
#     document = file.read() 

# store.build_tree(document)

# print(store.suggest_ucs('th'))  

