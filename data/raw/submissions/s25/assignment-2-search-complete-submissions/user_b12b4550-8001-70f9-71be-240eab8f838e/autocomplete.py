from collections import deque
import heapq

class Node:
    def __init__(self):
        self.children = {}  
        self.is_word = False  
        self.frequency = 0  

class Autocomplete:
    def __init__(self, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs  
        if document:
            self.build_tree(document)

    def build_tree(self, document):
        
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
                node.frequency += 1  
            node.is_word = True  
    def suggest_bfs(self, prefix):
       
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            current_node, path = queue.popleft()
            if current_node.is_word:
                suggestions.append(path)

            for char, child_node in current_node.children.items():
                queue.append((child_node, path + char))

        return suggestions

    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # 
            node = node.children[char]

        stack = [(node, prefix)]
        suggestions = []

        while stack:
            current_node, path = stack.pop()
            if current_node.is_word:
                suggestions.append(path)

            for char, child_node in sorted(current_node.children.items(), reverse=True):
                stack.append((child_node, path + char))

        return suggestions

    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # Prefix not found
            node = node.children[char]

        pq = [(0, node, prefix)]  # Priority Queue: (cost, node, word)
        suggestions = []

        while pq:
            cost, current_node, path = heapq.heappop(pq)
            if current_node.is_word:
                suggestions.append(path)

            for char, child_node in current_node.children.items():
                heapq.heappush(pq, (cost + 1 / child_node.frequency, child_node, path + char))

        return suggestions

    def print_tree(self, node=None, prefix=""):
        if node is None:
            node = self.root
        if node.is_word:
            print(prefix)
        for char, child in node.children.items():
            self.print_tree(child, prefix + char)
