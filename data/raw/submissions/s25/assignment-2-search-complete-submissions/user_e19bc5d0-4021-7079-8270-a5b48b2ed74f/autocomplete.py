from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_goal = False
        self.freq = {}
        # self.is_word = False

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
                    node.freq[char] = 1
                else:
                    node.freq[char] = node.freq[char] + 1
                node = node.children[char]
            node.is_goal = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestion_list = []
        # follow the prefix to get the subtree we are working with
        node = self.root
        for char in prefix:
            # if the prefix is not a prefix for any word in our dictionary, return []
            if char not in node.children:
                return suggestion_list
            # otherwise update our node to the child node for that character
            node = node.children[char]

        # perform bfs on that subtree, adding a word to the suggestion list when reaching a goal node
        queue = deque([(prefix, node)])
        while len(queue) != 0:
            (cur_prefix, cur_node) = queue.popleft() # CHOOSE

            # CHECK: if the cur_node is a goal, it is a word so add to list
            if cur_node.is_goal:
                suggestion_list.append(cur_prefix)
            
            # EXPAND AND ENQUEUE
            for c, c_node in cur_node.children.items():
                queue.append((cur_prefix + c, c_node))

        return suggestion_list

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestion_list = []
        # follow the prefix to get the subtree we are working with
        node = self.root
        for char in prefix:
            # if the prefix is not a prefix for any word in our dictionary, return []
            if char not in node.children:
                return suggestion_list
            # otherwise update our node to the child node for that character
            node = node.children[char]
        
        # perform dfs on that subtree, adding a word to the suggestion list when reaching a goal node
        stack = deque([(prefix, node)])
        while len(stack) != 0:
            (cur_prefix, cur_node) = stack.pop() # CHOOSE

            # CHECK: if the cur_node is a goal, it is a word so add to list
            if cur_node.is_goal:
                suggestion_list.append(cur_prefix)

            # EXPAND AND ENQUEUE
            for c, c_node in cur_node.children.items():
                stack.append((cur_prefix + c, c_node))

        return suggestion_list


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestion_list = []
        # follow the prefix to get the subtree we are working with
        node = self.root
        for char in prefix:
            # if the prefix is not a prefix for any word in our dictionary, return []
            if char not in node.children:
                return suggestion_list
            # otherwise update our node to the child node for that character
            node = node.children[char]

        # the path cost is the inverse of the frequency of words that start the prefix
        # UCS
        priority_queue = []
        heapq.heappush(priority_queue, (0, (prefix, node)))
        while len(priority_queue) != 0:
            cur_cost, (cur_prefix, cur_node) = heapq.heappop(priority_queue) # CHOOSE

            # CHECK if goal
            if cur_node.is_goal:
                suggestion_list.append(cur_prefix)

            # EXPAND AND ENQUEUE
            for c, c_node in cur_node.children.items():
                # get the next nodes cost and add it to the current total path cost
                cost = cur_cost + 1/cur_node.freq[c]
                heapq.heappush(priority_queue, (cost, (cur_prefix + c, c_node)))
        
        return suggestion_list
