from collections import deque
import heapq
import random
import string


class Node:
    # TODO
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.suggest = self.suggest_ucs

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                # TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_end_of_word = True  
            # pass

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(
            string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    # TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  

        suggestions = []
        queue = deque([(node, prefix)])


        def bfs():
            while queue:
                current_node, current_prefix = queue.popleft()
                if current_node.is_end_of_word:
                    suggestions.append(current_prefix)

                for child_char, child_node in current_node.children.items():
                    queue.append((child_node, current_prefix + child_char))

        bfs()  
        return suggestions

    # TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []  
        suggestions = []
        stack = [(node, prefix)]
        while stack:
            current_node, current_prefix = stack.pop()  
            if current_node.is_end_of_word:
                suggestions.append(current_prefix)
            for child_char, child_node in current_node.children.items():
                stack.append((child_node, current_prefix + child_char))

        return suggestions

    # TODO for students!!!

    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return [] 
        suggestions = []
        priority_queue = [] 

        def add_to_queue(node, current_prefix, cost):
            if node.is_end_of_word:
                heapq.heappush(priority_queue, (cost, current_prefix))

            for child_char, child_node in node.children.items():
                add_to_queue(child_node, current_prefix +
                             child_char, cost + 1) 

        add_to_queue(node, prefix, 0)

        while priority_queue:
            _, word = heapq.heappop(priority_queue)
            suggestions.append(word)

        return suggestions
