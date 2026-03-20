from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.path_cost = 1

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.


    # Iterates through each letter in each word in the document, if there is not already
    # a node for that character at that level of the tree creates one and adds it to the children of the node for the
    # prior letter in the word, or the root. Otherwise it updates the edge cost from the previous node to that node.
    # If the character is the final letter of a word sets the nodes is_word attribute to true.
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                # Creates the nodes for each char under the previous char if that char is not already
                # in the children list for the previous char
                if char not in node.children:
                    node.children[char] = Node()
                else:
                    cost = node.children[char].path_cost
                    cost = 1/cost
                    cost = 1 + cost
                    node.children[char].path_cost = 1/cost
                node = node.children[char]
            node.is_word = True # Signifies this letter is the end of a word



    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    # Runs a BFS starting at the final character of the prefix. Appends a tuple of the current node seen in
    # the BFS as well as the substring up to that node to a queue. When a node has been explored and removed from the front
    # of the queue it checks if that node is the end of a word and if it is adds the string in the tuple with that node to the suggestions array
    # that is returned. It then adds all of the tuples for the explored node's children to the queue, appending their letter to the substring in their tuple.
    def suggest_bfs(self, prefix):
        q = deque()
        node = self.root
        suggestions = []

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        q.append((node, prefix))

        while q:
            current_node, current_str = q.popleft()

            if current_node.is_word:
                suggestions.append(current_str)

            for char in current_node.children:
                new_str = current_str + char
                new_node = current_node.children[char]

                q.append((new_node, new_str))

        return suggestions

    #TODO for students!!!
    # Runs a DFS starting at the final character of the prefix. Appends a tuple of the current node seen in
    # the DFS as well as the substring up to that node to a stack. When a node has been explored and removed from the top
    # of the stack it checks if that node is the end of a word and if it is adds string in the tuple with that node to the suggestions array
    # that is returned. It then adds all of the tuples for the explored node's children to the stack, appending their letter to the substring in their tuple.
    def suggest_dfs(self, prefix):
        q = deque()
        node = self.root
        suggestions = []

        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                return []

        q.append((node, prefix))

        while q:
            current_node, current_str = q.pop()

            if current_node.is_word:
                suggestions.append(current_str)

            for char in current_node.children:
                new_str = current_str + char
                new_node = current_node.children[char]

                q.append((new_node, new_str))

        return suggestions

    #TODO for students!!!
    # Appends a triple of the current node seen in the UCS as well as the substring up to that node and that node's path
    # cost to a priority queue ordered by smallest path_cost first. When a node has been explored and removed from the top
    # of the stack it checks if that node is the end of a word and if it is adds the word to the suggestions array
    # that is returned. It then adds the explored node's path_cost to the path_cost of the node's children and adds the
    # triples for each child to the priority queue.
    def suggest_ucs(self, prefix):
        node = self.root
        suggestions = []

        for char in prefix:
            if char in node.children:
                node.children[char].path_cost += node.path_cost
                node = node.children[char]
            else:
                return []

        h = [(node.path_cost, prefix, node)]

        while h:
            _, current_str, current_node = heapq.heappop(h)

            if current_node.is_word:
                suggestions.append(current_str)

            for char in current_node.children:
                new_str = current_str + char
                new_node = current_node.children[char]
                new_node.path_cost += current_node.path_cost

                heapq.heappush(h, (new_node.path_cost, new_str, new_node))

        return suggestions
