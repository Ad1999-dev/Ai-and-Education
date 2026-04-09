from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.cost = 0
        self.is_word_end = False
        self.children = {}

    # Convert class Node to string form by combining its cost with
    # children characters
    def __str__(self):
        keys = list(self.children.keys())
        return f'{str(self.cost)}: {",".join(keys)}: {str(self.is_word_end)}'


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        # self.suggest = self.suggest_random
        self.suggest = self.suggest_bfs

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            # keep track of the current char's index in word
            counter = 0
            for char in word:
                # If char is not a child of node, make it a
                # child of node and assign it a cost of 1
                if char not in node.children:
                    node.children[char] = Node()
                    node.children[char].cost = 1
                else:
                    # char is an existing child of node,
                    # reassign its cost
                    node.children[char].cost = 1 / ((1 / node.children[char].cost) + 1)

                # mark the node as the end of a word
                counter += 1
                node.children[char].is_word_end = node.children[char].is_word_end or counter == len(word)
                node = node.children[char]

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    # TODO for students!!!
    def suggest_bfs(self, prefix):
        if self.root is None:
            return []
        result = []
        queue = [(self.root, '')]

        while len(queue) > 0:
            # dequeue
            tu = queue.pop(0)
            node = tu[0]
            word = tu[1]

            if word.startswith(prefix) and node.is_word_end:
                result.append(word)

            for child_key in node.children.keys():
                new_word = word + child_key
                queue.append((node.children[child_key], new_word))

        # for w in result:
        #     print(w)
        #     print()

        return result

    # TODO for students!!!
    def suggest_dfs(self, prefix):
        def recursive():
            if self.root is None:
                return []

            result = []

            def dfs(node, curr_word):
                if node is None:
                    return

                for child_key in node.children.keys():
                    child_node = node.children[child_key]
                    new_word = curr_word + child_key
                    if new_word.startswith(prefix) and node.children[child_key].is_word_end:
                        result.append(new_word)

                    dfs(child_node, new_word)

            dfs(self.root, '')

            # for w in result:
            #     print(w)
            #     print()

            return result

        def iterative():
            if self.root is None:
                return []
            result = []
            queue = [(self.root, '')]

            while len(queue) > 0:
                # dequeue
                tu = queue.pop()
                node = tu[0]
                word = tu[1]

                if word.startswith(prefix) and node.is_word_end:
                    result.append(word)

                for child_key in reversed(node.children.keys()):
                    new_word = word + child_key
                    queue.append((node.children[child_key], new_word))
            return result

        return recursive()

    # TODO for students!!!
    def suggest_ucs(self, prefix):
        if self.root is None:
            return []

        result = []
        # total_cost, curr_word, node
        lst = [(0, '', self.root)]
        heapq.heapify(lst)

        while len(lst) > 0:
            # dequeue
            tu = heapq.heappop(lst)
            total_cost = tu[0]
            word = tu[1]
            node = tu[2]

            if node.is_word_end and word.startswith(prefix):
                result.append(word)

            for child_key in node.children.keys():
                new_word = word + child_key
                cost = node.children[child_key].cost
                total_cost_with_cost = total_cost + cost
                heapq.heappush(lst, (total_cost_with_cost, new_word, node.children[child_key]))

        # for w in result:
        #     print(w)
        #     print()

        return result
