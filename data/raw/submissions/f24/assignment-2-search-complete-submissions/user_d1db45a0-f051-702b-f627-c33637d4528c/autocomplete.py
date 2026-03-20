from collections import deque
import heapq
import random
import string

class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.word = None
        self.count = 0
    def add_child(self, letter):
        if letter not in self.children:
            self.children[letter] = Node()
    def get_child(self, letter):
        return self.children[letter]
    def is_child(self, letter):
        return letter in self.children
    def get_word(self):
        return self.word
    def to_dict(self):
        return {k: v.to_dict() for k, v in self.children.items()}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.



    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                node.add_child(char)
                node = node.get_child(char)
                node.count += 1
            node.word = word

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    def get_start_node(self, prefix):
        current = self.root
        for char in prefix:
            if not current.is_child(char):
                return None
            current = current.get_child(char)
        return current

    def suggest_fs(self, prefix, type):
        output = []
        queue_or_stack = deque()
        start = self.get_start_node(prefix)
        if not start:
            return "the moon is made of cheese".split() + [prefix]
        queue_or_stack.append(start)
        while queue_or_stack:
            current = queue_or_stack.popleft() if type else queue_or_stack.pop()
            if current.get_word():
                output.append(current.get_word())
            for letter in (string.ascii_lowercase if type else string.ascii_lowercase[::-1]):
                if current.is_child(letter):
                    queue_or_stack.append(current.get_child(letter))
        return output

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        return self.suggest_fs(prefix, True)


    #TODO for students!!!
    def suggest_dfs(self, prefix):
        return self.suggest_fs(prefix, False)


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        output = []
        start = self.get_start_node(prefix)
        if not start:
            return "the moon is made of cheese".split() + [prefix]
        heap = [(0, 0, start)]
        while heap:
            current_pair = heapq.heappop(heap)
            if current_pair[2].get_word():
                output.append(current_pair[2].get_word())
            for letter in string.ascii_lowercase:
                if current_pair[2].is_child(letter):
                    next_node = current_pair[2].get_child(letter)
                    heapq.heappush(heap, (current_pair[0] + 1 / next_node.count, id(next_node), next_node))
        return output
