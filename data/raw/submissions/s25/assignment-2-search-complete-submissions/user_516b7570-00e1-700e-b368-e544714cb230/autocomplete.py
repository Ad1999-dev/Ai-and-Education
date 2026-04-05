from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_end_word = False
        self.weight = 0

    def __lt__(self, other): # Comparing weights for UCS
        return self.weight < other.weight

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    # ------------------ BUILD TREE ---------------------
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                # This code will be implementing a Trie data structure

                if char not in node.children:
                    node.children[char] = Node()

                node = node.children[char]
                node.weight += 1 # Changes the weight higher for every charecter in the children char
            
            node.is_end_word = True 

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #---------------- BFS SEARCH ----------------
    def suggest_bfs(self, prefix):

        node = self.root

        for a in prefix:
            if not node.children.get(a): return []
            node = node.children[a]
        if not node.children: return [prefix]

        suggestions = []
        q = deque()
        q.append((node, prefix))

        while q:
            n, p = q.popleft()
            if n.is_end_word:
                suggestions.append(p)
            for a, n in n.children.items():
                q.append((n, p + a))
        return suggestions


    # ------------------  DFS SEARCH --------------------------
    def suggest_dfs(self, prefix):
        node = self.root

        # Traverse the Trie according to the prefix
        for a in prefix:
            if not node.children.get(a):
                return []  # If prefix is not found, return empty list
            node = node.children[a]

        # If we reached a node but it has no children, we can still return the prefix if it is an end word
        suggestions = self.get_dfs_word(node, prefix)

        return suggestions

    def get_dfs_word(self, node, prefix):
        suggestions = []

        # If the current node marks the end of a word, add the prefix to suggestions
        if node.is_end_word:
            suggestions.append(prefix)

        # Explore all child nodes
        for a, n in node.children.items():
            # Accumulate suggestions from child nodes
            suggestions.extend(self.get_dfs_word(n, prefix + a))

        return suggestions

    # ------------------ UCS SEARCH --------------------------------
    def suggest_ucs(self, prefix):
        node = self.root

        for a in prefix:
            if not node.children.get(a):
                return []
            node = node.children[a] 

        if not node.children:
            return [prefix]

        ans = []
        priorityq = []
        heapq.heappush(priorityq, (node.weight, node, prefix))

        while priorityq:
            currentcost, current_node, current_prefix = heapq.heappop(priorityq)

            if current_node.is_end_word:
                ans.append(current_prefix)

            for a, n in current_node.children.items():
                heapq.heappush(priorityq, (currentcost + (float(1)/n.weight), n, current_prefix + a))

        return ans
