from collections import deque, defaultdict
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.freq = defaultdict(int)

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node.freq[char] += 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            current_node, current_word = queue.popleft()

            if current_node.is_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                queue.append((child_node, current_word + char))

        return suggestions

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        stack = [(node, prefix)]
        suggestions = []

        while stack:
            current_node, current_word = stack.pop()

            if current_node.is_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                stack.append((child_node, current_word + char))
        
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        heap = [(0, prefix, node)]
        suggestions = []

        while heap:
            freq, current_word, current_node = heapq.heappop(heap)

            if current_node.is_word:
                suggestions.append(current_word)

            for char, child_node in current_node.children.items():
                heapq.heappush(heap, (node.freq[char], current_word + char, child_node))

        return suggestions
    
autocomplete = Autocomplete()

# Read the document file and build the tree
with open("genZ.txt", "r") as f:
    text_data = f.read()

autocomplete.build_tree(text_data)

# Generate autocomplete suggestions for prefix "th"
suggestions_bfs = autocomplete.suggest_bfs("th")
suggestions_dfs = autocomplete.suggest_dfs("th")
suggestions_ucs = autocomplete.suggest_ucs("th")

# priting
print(suggestions_bfs)  #
print(suggestions_dfs)  #
print(suggestions_ucs)  # 