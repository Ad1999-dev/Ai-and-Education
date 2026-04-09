from collections import deque
import heapq
import random
import string


def get_start(root, prefix):
    cur_cost = 0
    for char in prefix:
        if char in root.children:
            cur_cost += 1 / root.child_frequencies[char]
            root = root.children[char]
    return root, cur_cost


def enqueue(node, prefix, suggestions, frontier):
    if node.is_word:
        suggestions.append(prefix)
    for char, child in node.children.items():
        frontier.append((child, prefix + char))


class Node:
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.child_frequencies = {}


class Autocomplete:
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                    node.child_frequencies[char] = 1
                else:
                    node.child_frequencies[char] += 1
                node = node.children[char]
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [
            "".join(random.choice(string.ascii_lowercase) for _ in range(3))
            for _ in range(5)
        ]
        return [prefix + suffix for suffix in random_suffixes]

    def suggest_bfs(self, prefix):

        frontier = deque([(get_start(self.root, prefix=prefix)[0], prefix)])
        suggestions = []

        while frontier:
            cur_node, cur_prefix = frontier.popleft()
            enqueue(cur_node, cur_prefix, suggestions, frontier)

        return suggestions

    def suggest_dfs(self, prefix):

        frontier = deque([(get_start(self.root, prefix=prefix)[0], prefix)])
        suggestions = []

        while frontier:
            cur_node, cur_prefix = frontier.pop()
            enqueue(cur_node, cur_prefix, suggestions, frontier)

        return suggestions

    def suggest_ucs(self, prefix):

        frontier = []
        start, start_cost = get_start(self.root, prefix=prefix)
        heapq.heappush(frontier, (start_cost, prefix, start))
        explored_set = {}
        suggestions = []

        while frontier:

            cur_cost, cur_prefix, cur_node = heapq.heappop(frontier)

            if cur_node in explored_set:
                if explored_set[cur_node] <= cur_cost:
                    continue

            explored_set[cur_node] = cur_cost

            if cur_node.is_word:
                suggestions.append(cur_prefix)
            for char, child in cur_node.children.items():
                prior_path_cost = cur_cost + 1 / cur_node.child_frequencies[char]   
                heapq.heappush(frontier, (prior_path_cost, cur_prefix + char, child))

        return suggestions
