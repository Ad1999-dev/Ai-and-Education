from collections import deque
import heapq
import pprint
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.letter = ''
        self.string = ""
        self.frequency = 0;



class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.string = ""
        self.letter = ""
        self.frequency = 0;

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                
                #TODO for students

                if not (char in node.children):
                    node.children[char] = Node()
                    node.children[char].letter = char
                node = node.children[char]
                node.frequency += 1

            node.string = word
            node.children["empty"] = Node()
            node.children["empty"].letter = ''


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        word_list = []
        queue = deque()
        visited = []
        
        node = self.root;
        queue.append(node)
        visited.append(node)
        while  (queue):
            node = queue.popleft()
            if (hasattr(node, 'string') and node.string != '' and node.string.startswith(prefix)):
                word_list.append(node.string)

            for child_node in node.children.values():
                if child_node not in visited:
                    visited.append(child_node)
                    queue.append(child_node)

        print(word_list)
        return word_list;

    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        word_list = []
        stack = [];
        visited = []

        node = self.root
        stack.append(node)

        while (len(stack) > 0):
            node = stack.pop()
            if node not in visited:
                if (hasattr(node, 'string') and node.string != '' and node.string.startswith(prefix)):
                    word_list.append(node.string)
                visited.append(node)
                for child_node in node.children.values():
                    stack.append(child_node)
                
        return word_list;


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        def get_cost_invert(node):
            frequency = node[1]
            if (frequency == 0):
                return 0
            else:
                return 1 / frequency
        visited = []
        frontier = []
        word_list = []
        node = self.root
        frontier.append([self.root, self.root.frequency])
        while (len(frontier) > 0):
            frontier.sort(reverse = False, key=get_cost_invert)
            frontier_item = frontier.pop(0)
            node = frontier_item[0]
            current_frequency = frontier_item[1]
            if (hasattr(node, 'string') and node.string != '' and node.string.startswith(prefix)):
                word_list.append(node.string)
            visited.append(node)
            for child_node in node.children.values():
                if child_node not in visited:
                    frontier.append([child_node, current_frequency * child_node.frequency])
        return word_list



    
