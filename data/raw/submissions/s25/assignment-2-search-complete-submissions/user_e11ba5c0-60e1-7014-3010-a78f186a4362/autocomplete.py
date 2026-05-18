from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self, prev=None, value="root", is_word=False):
        self.value = value
        self.children = dict()
        self.is_word = is_word
        self.prev = prev
        self.seen = 0
        self.cost = 1
    
    def see(self):
        self.seen += 1
        self.cost = 1 / self.seen

    def __lt__(self, other):
        return self.cost < other.cost


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        # Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
        self.suggest = self.suggest_bfs 
    
    #
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            node.see()
            for i, char in enumerate(word):
                if char not in node.children:
                    node.children[char] = Node(prev=node, value=char)
                node = node.children[char]
                node.see()
                # is_word is true if this is the last letter in the word
                node.is_word = node.is_word or (i == len(word) - 1)
            

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def get_prefix_end_node(self, prefix):
        node = self.root
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                # prefix has no end in the document so there is no end node
                return []
        return node

    def build_word(self, node: Node):
        word = ""
        while node.prev is not None:
            word = node.value + word
            node = node.prev
        return word

    # search requires:
    # observable, discrete, known, deterministic environment

    # CHECK, EXPAND, ENQUEUE, CHOOSE
    # In this case, there is no goal so we don't need to check
    def suggest_search(self, prefix, search_type):

        node = self.get_prefix_end_node(prefix)
        if node == []: return []
        # now the node is the last letter of the prefix

        suggestions = []

        if search_type in ["bfs", "dfs"]:
            frontier = deque()
            frontier.append(node) 
        elif search_type == "ucs":
            frontier = [(0, node)]

        while frontier:
            # in the non-recursive version, the only difference between 
            # bfs and dfs in a tree is which order we take from the frontier
            if search_type == "bfs":
                node = frontier.popleft()
                frontier.extend(node.children.values())
            elif search_type == "dfs":
                node = frontier.pop()
                frontier.extend(node.children.values())
            elif search_type == "ucs":
                cost, node = heapq.heappop(frontier)

                for child in node.children.values():
                    heapq.heappush(frontier, (cost + child.cost, child))
            else:
                node = frontier.pop()
                frontier.extend(node.children.values())
            
            if node.is_word:
                suggestions.append(self.build_word(node))
        
        return suggestions

    def suggest_bfs(self, prefix):
        return self.suggest_search(prefix, "bfs")

    def suggest_dfs(self, prefix):
        return self.suggest_search(prefix, "dfs")

    def suggest_ucs(self, prefix):
        return self.suggest_search(prefix, "ucs")