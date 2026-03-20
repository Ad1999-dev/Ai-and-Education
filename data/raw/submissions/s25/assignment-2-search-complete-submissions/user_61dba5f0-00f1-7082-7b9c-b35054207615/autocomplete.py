from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word_count = 0  # Track number of words that can be formed from this node

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:

                # if the character is not any of the children of the root node, then create a new node starting with that char
                if char not in node.children:
                    node.children[char] = Node()
                
                # move the current node to be the node that represents the current char
                node = node.children[char]
                node.word_count += 1  # Increment word count at each node
            # mark this node as the end of a word
            node.is_word = True
            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        suggestions = []

        # go to the node where the prefix ends
        for i in prefix:
            if i in node.children:
                node = node.children[i]
            else:
                return [] #no suggestion if prefix is not found in the tree
        
        queue = deque([(node, prefix)])  #using a queue for bfs


        while queue:
            current_node, current_prefix = queue.popleft()  # Dequeue the next node

            if current_node.is_word:
                suggestions.append(current_prefix)  # Add the word to suggestions if its a word

            for char, child_node in current_node.children.items():
                queue.append((child_node, current_prefix + char)) # Enqueue child nodes.
        return suggestions

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        suggestions = []

        # go to the node where the prefix ends
        for i in prefix:
            if i in node.children:
                node = node.children[i]
            else:
                return [] #no suggestion if prefix is not found in the tree
        
        stack = [(node, prefix)]  # using a stack based dfs
        while stack:
            current_node, current_prefix = stack.pop() # Pop the next node from the stack

            
            if current_node.is_word:
                suggestions.append(current_prefix)  # Add the word to suggestions if its a word.

            
            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))  # Push child nodes onto the stack.

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        suggestions = []

        # go to the node where the prefix ends
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] #no suggestion if prefix is not found in the tree
            
        # Priority queue (cost, char, node, prefix) to enforce alphabetical order on ties
        priority_queue = [(0, "", node, prefix)]  # using priority queue for ucs

        while priority_queue:
            cost, char, current_node, current_prefix = heapq.heappop(priority_queue)

            if current_node.is_word:
                suggestions.append(current_prefix)

            for next_char in sorted(current_node.children.keys()):  # Sorting to make sure its in alphabetical order
                child_node = current_node.children[next_char]
                if child_node.word_count > 0:
                    new_cost = cost + (1 / child_node.word_count) #calculating cost by adding the inverse number of words that come from that node
                    heapq.heappush(priority_queue, (new_cost, next_char, child_node, current_prefix + next_char))

        return suggestions
