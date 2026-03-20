from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, char):
        self.children = {}
        self.freq = {}
        self.seen = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node('Root')
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        #tracker for already visited nodes/words
        #track the frequency of character/word occurrence    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            curr_word = ''
            for char in word:
                curr_word += char
                if (char not in node.children):
                    node.children[char] = Node(char) # creates child
                    node.freq[char] = 1
                else:
                    node.freq[char]+=1
                node = node.children[char] # goes to child, becomes current node (happens regardless if the child is created or not)             
            node.seen = True
                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):

        # find node corresponding to last char in prefix
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return[] #prefix not in the tree

        # perform BFS
        suggestions = []
        queue = deque([(node, prefix)])
        while queue:
            curr_node, curr_prefix = queue.popleft()
            
            # if curr node is leaf node
            if not curr_node.children or curr_node.seen: #end of word
                suggestions.append(curr_prefix) #should i end at leaf node?
            
            # add all children to queue
            for char, child in curr_node.children.items():
                # enqueue child node and current word/prefix
                queue.append((child, curr_prefix + char))
        return suggestions

    #TODO for students!!!

    # find the last node
    def suggest_dfs(self, prefix):
        node = self.root

        # same as bfs, find the last char in the prefix by traversing the tree for that node
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        suggestions = []

        # create a helper function to perform dfs recursively
        # this func works similar to bfs but instead of adding neighbors to queue, we call the dfs function on the next child node and current word building upon the prefix
        def dfs_helper(curr_node, curr_prefix):
            if not curr_node.children or curr_node.seen:
                suggestions.append(curr_prefix)

            for char, child in curr_node.children.items():
                dfs_helper(child, curr_prefix+char)
        
        dfs_helper(node, prefix)
        return suggestions

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # goal: Choose the node with the least total path cost until you reach the goal
        
        node = self.root

        #perform tree traversal to find the last char-node associated with prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []
        
        suggestions = []
        priority_queue = [] # for ucs, we will be using a priority queue to explore nodes based on their cumulative "cost"
        heapq.heappush(priority_queue, ((0, prefix, node))) #tuple now has 3 values instead of 2 because of "cost"

        while priority_queue:
            cost, curr_prefix, curr_node = heapq.heappop(priority_queue) #3 vars are defined locally now

            if not curr_node.children or curr_node.seen:
                # suggestions.append(curr_prefix)
                heapq.heappush(suggestions,(cost, curr_prefix))

            for char, child in curr_node.children.items():
                print((char,curr_node.freq[char]))
                new_cost = cost + (1/(curr_node.freq[char]))
                heapq.heappush(priority_queue, (new_cost, curr_prefix+char, child))

        return [heapq.heappop(suggestions)[1] for _ in range(len(suggestions))]
        
