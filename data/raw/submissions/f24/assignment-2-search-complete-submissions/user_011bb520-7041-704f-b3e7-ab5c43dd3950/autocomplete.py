from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, cost):
        self.children = {}
        self.end = False
        self.cost = cost

    def calc_cost(self):
        for key, child in self.children.items():
            self.cost = len(self.children)
            child.calc_cost()

    def __lt__(self, other):
        return self.cost < other.cost


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node(1)
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` suggest_random based on which one you wish to use.
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            word_length = len(word)
            for index, char in enumerate(word):
                if char in node.children:
                    node = node.children[char]
                else:
                    new_node = Node(1)
                    node.children[char] = new_node
                    node = new_node
                node.cost += 1
                if (index == word_length - 1):
                    node.end = True


    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    def suggest_bfs(self, prefix):
        node = self.root
        results = []
        visited = set()

        # Traverse down the tree to find the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return results  # Prefix is not found, return empty list

        # Now we perform BFS from this node
        queue = deque([(node, prefix)])  # Queue of (node, current prefix)
        
        while queue:
            current_node, current_prefix = queue.popleft() # Pop from the left for BFS
            
            # If this node has been visited before, skip
            if current_node in visited:
                continue
            visited.add(current_node)

            # If this node marks the end of a word, add to results
            if current_node.end:
                results.append(current_prefix)

            # Enqueue all children nodes
            for char, child_node in current_node.children.items():
                if child_node not in visited:
                    queue.append((child_node, current_prefix + char))

        return results

    def suggest_dfs(self, prefix):
        node = self.root
        results = []
        visited = set()

        # Traverse down the tree to find the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return results  # Prefix is not found, return empty list

        # Now we perform DFS from this node
        stack = [(node, prefix)]  # Stack of (node, current prefix)

        while stack:
            current_node, current_prefix = stack.pop()  # Pop from the stack for DFS

            # If this node has been visited before, skip
            if current_node in visited:
                continue
            visited.add(current_node)
            
            # If this node marks the end of a word, add to results
            if current_node.end:
                results.append(current_prefix)

            # Push all children nodes onto the stack
            for char, child_node in current_node.children.items():
                if child_node not in visited:
                    stack.append((child_node, current_prefix + char))

        return results
    

    def suggest_ucs(self, prefix):
        node = self.root
        results = []
        visited = set()

        # Traverse down the tree to find the prefix
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return results  # Prefix is not found, return empty list
            
        # Now we perform UCS from this node
        queue = [(0, node, prefix)]  # Queue of (node, current prefix)


        while queue:
            current_cost, current_node, current_prefix = heapq.heappop(queue)

            # If this node has been visited before, skip
            if current_node in visited:
                continue
            visited.add(current_node)

            # If this node marks the end of a word, add to results
            if current_node.end:
                results.append(current_prefix)

            # Enqueue all children nodes
            for char, child_node in current_node.children.items():
                if child_node not in visited:
                    heapq.heappush(queue, (current_cost + 1 / child_node.cost ,child_node, current_prefix + char))

        return results
        
