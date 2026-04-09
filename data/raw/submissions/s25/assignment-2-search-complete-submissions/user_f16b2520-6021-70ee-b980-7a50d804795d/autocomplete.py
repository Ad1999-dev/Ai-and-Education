from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, char):
        self.value = char
        self.children = {}
        self.is_word = False
        self.frequency = 1
    def addFrequency(self):
        self.frequency +=1
        print("added 1 to frequency of " + self.value)
    def __lt__(self, other):
        return self.frequency > other.frequency
    def setCurrentWord(self, input):
        self.current_word = input

class Autocomplete():
    def __init__(self, parent=None, document=""):
        nullChar = '\0'
        self.root = Node(nullChar)
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        words_list = document.split()
        unique_words = list(dict.fromkeys(words_list))
        for word in unique_words:
            node = self.root
            for char in word:
                #TODO for students
                if char in node.children:
                    node = node.children[char]
                    node.addFrequency()
                else:
                    nextChar = Node(char)
                    node.children[char] = nextChar
                    node = nextChar
            node.is_word = True

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        words = []
        node = self.root
        prefixExists = True
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                prefixExists = False
                break
        if prefixExists:
            queue = deque([(node, prefix)])
            while queue:
                node, wordPath = queue.popleft()
                if node.is_word:
                    words.append(wordPath)
                for child in node.children:
                    queue.append((node.children[child], wordPath + node.children[child].value))
        return words


    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        words = []
        node = self.root
        prefixExists = True
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                prefixExists = False
                break
        if prefixExists:
            queue = deque([(node, prefix)])
            while queue:
                node, wordPath = queue.popleft()
                if node.is_word:
                    words.append(wordPath)
                for child in node.children:
                    queue.appendleft((node.children[child], wordPath + node.children[child].value))
        return words


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        print("beginning ucs")
        words = []
        node = self.root
        prefixExists = True
        for char in prefix:
            if char in node.children:
                node = node.children[char]
            else:
                prefixExists = False
                break
        if prefixExists:
            node.setCurrentWord(prefix)
            queue = []
            heapq.heappush(queue, node)
            while queue:
                current_node = heapq.heappop(queue)
                print("Exploring node " + current_node.value + " with word path of " + current_node.current_word + " and frequency " + str(current_node.frequency))
                if current_node.is_word:
                    words.append(current_node.current_word)
                for child in current_node.children.values():
                    child.setCurrentWord(current_node.current_word + child.value)
                    heapq.heappush(queue, child)
                    print("Added " + child.value + " to queue with frequency " + str(child.frequency))
        return words
