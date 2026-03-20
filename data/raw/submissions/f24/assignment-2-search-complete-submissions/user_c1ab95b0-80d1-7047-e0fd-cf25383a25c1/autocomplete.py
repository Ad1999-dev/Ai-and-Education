from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.cost = {}
        self.end_of_word = False

class Autocomplete():
    def __init__(self, parent=None, document=""):
        print("Initializing Autocomplete")
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def build_tree(self, document):
        print("Building Tree")
        frequencies = {}
        for word in document.split():
            print(word)
            node = self.root
            cumulative_cost = 0
            for char in word:
                if char not in frequencies:
                    frequencies[char] = 0
                frequencies[char] += 1
                
                if char not in node.children:
                    node.children[char] = Node()
                    print(f"Creating node for character: '{char}'")
                    
                cumulative_cost += 1 / frequencies[char]
                node = node.children[char]
                node.cost = cumulative_cost
                
                print(f"Traversed: '{char}' with cumulative cost: {node.cost}")
            node.end_of_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        print("\nBFS STARTING\n")
        node = self.root
        q = deque()

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        q.append((node, prefix))
        words = []

        while q:
            current_node, current_prefix = q.popleft()

            if current_node.end_of_word:
                words.append(current_prefix)

            for char, child_node in current_node.children.items():
                q.append((child_node, current_prefix + char))

        # print(words)
        return words

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        print("\nDFS STARTING\n")
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        stack = [(node, prefix)]
        words = []

        while stack:
            current_node, current_prefix = stack.pop()

            if current_node.end_of_word:
                words.append(current_prefix)

            for char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + char))

        return words


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        print("\nUCS STARTING\n")
        node = self.root
        q = deque()

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        priorityqueue = []
        heapq.heappush(priorityqueue, (0, node, prefix))
        words = []

        while priorityqueue:
            current_cost, current_node, current_prefix = heapq.heappop(priorityqueue)

            if current_node.end_of_word:
                words.append(current_prefix)
                #print(f"Found suggestion: {current_prefix} with cost: {current_cost}")

            for char, child_node in current_node.children.items():
                new_cost = current_cost + child_node.cost
                heapq.heappush(priorityqueue, (new_cost, child_node, current_prefix + char))
                #print(f"Pushing to queue: {current_prefix + char} with cost: {new_cost}")

        return words
