from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end_of_word = True
    
    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char] #Traverses to the next letter
            else:
                return [] #if no words with that prefix exist
        
        #BFS
        suggestions = []
        queue = deque([[node, prefix]])
       
        
        while queue:
            current_node, word = queue.popleft() #BFS pops from the left
            if current_node.is_end_of_word:
                suggestions.append(word)
            
            for char, next_node in sorted(current_node.children.items()):
                queue.append((next_node, word + char))
        print(suggestions)
        return suggestions
    
    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        stack = deque([[node, prefix]])
        suggestions = []
        
        while stack:
            current_node, word = stack.pop() #DFS pops from bottom
            if current_node.is_end_of_word:
                suggestions.append(word)
                
            for char in sorted(current_node.children.keys(), reverse=True):
                next_node = current_node.children[char]
                stack.append((next_node, word + char))
                
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
    
    # Traverse the trie to find the end node of the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  # If the prefix doesn't exist, return an empty list

    # Priority queue for UCS: (cost, node, word)
        priority_queue = []
        heapq.heappush(priority_queue, (0, node, prefix))  # Start with cost 0
        suggestions = []

        while priority_queue:
            cost, current_node, word = heapq.heappop(priority_queue)  # Unpack correctly

            if current_node.is_end_of_word:  # Check if it is a valid word
                suggestions.append(word)
        
        # Iterate through child nodes
            for char, next_node in current_node.children.items():
                new_cost = cost + 1  # Assuming a uniform cost for each character
            # Push onto priority queue without causing comparisons on Node instances
                heapq.heappush(priority_queue, (new_cost, next_node, word + char))

            return suggestions  # Return the list of suggestions found


  
  
  ##Main for printing the diagram!
    # def main(): #Making the Diagram
    #     with open('test.txt', 'r') as file:
    #         content = file.read()
    #         tree = Node()
    #         tree.build_tree(content)
            
    #         print("This is the Tree of test.txt")
            
    # if __name__ == "__main__":
    #     ac = Autocomplete()
    #     print(ac.su)