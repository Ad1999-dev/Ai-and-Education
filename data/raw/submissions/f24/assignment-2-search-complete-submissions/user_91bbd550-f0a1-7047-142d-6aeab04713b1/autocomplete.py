from collections import deque
import heapq
import random
import string


class Node:
   #TODO
   def __init__(self):
       self.children = {}
       self.is_end_of_word = False
       self.char_frequency = {}


class Autocomplete():
   def __init__(self, parent=None, document=""):
       self.root = Node()
       self.suggest = self.suggest_ucs #self.suggest_random #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


  
  
   def build_tree(self, document):
       #words are being split up. Ex: "hello word" --> ["hello", "world"]. Splits the document into individual words.
       for word in document.split():
           node = self.root
           for char in word:
               #TODO for students
               #If node with the character doesn't exists --> Create a new node with that character. Now our node is pointing to the new node created.
               if(char not in node.children):
                   #Create new Node
                   node.children[char] = Node()
               if char not in node.char_frequency:
                    node.char_frequency[char] = 0
               node.char_frequency[char] += 1
               node = node.children[char]
           node.is_end_of_word = True


   def suggest_random(self, prefix):
       random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
       return [prefix + suffix for suffix in random_suffixes]


   #TODO for students!!!
   def suggest_bfs(self, prefix):
       node = self.root
       for char in prefix:
           if(char not in node.children):
               return []
           node = node.children[char]

       fifo_queue = deque([(node, prefix)])
       suggestions = []

       while fifo_queue:
            current_node, current_prefix = fifo_queue.popleft()
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)
            for char, child_node in current_node.children.items():
                fifo_queue.append((child_node, current_prefix + char))

       return suggestions




   #TODO for students!!!
   def suggest_dfs(self, prefix):
       node = self.root
       for char in prefix:
           if(char not in node.children):
               return []
           node = node.children[char]   

       lifo_queue = deque([(node, prefix)])
       suggestions = []

       while lifo_queue:
            current_node, current_prefix = lifo_queue.pop()
            if current_node.is_end_of_word:
                 suggestions.append(current_prefix)

            for char in sorted(current_node.children.keys(), reverse=True):
                 child_node = current_node.children[char]
                 lifo_queue.append((child_node, current_prefix + char))
           
        
       return suggestions



   #TODO for students!!!

   def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  
            node = node.children[char]

        pq = [] 
        suggestions = []
        counter = 0 

        heapq.heappush(pq, (0, counter, node, prefix))

        while pq:
            cost, _, current_node, current_prefix = heapq.heappop(pq)

            if current_node.is_end_of_word:
                suggestions.append(current_prefix)

            for char, child_node in current_node.children.items():
                if char in current_node.char_frequency:
                    char_freq = current_node.char_frequency.get(char, 1) 
                    path_cost = cost + (1 / char_freq)  

                    counter += 1

                    heapq.heappush(pq, (path_cost, counter, child_node, current_prefix + char))

        return suggestions

       
