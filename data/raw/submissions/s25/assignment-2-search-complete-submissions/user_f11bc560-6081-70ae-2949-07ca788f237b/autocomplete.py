from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, char=None, parent=None, cost=0):
        self.children = {}
        self.is_word = False
        self.parent = parent
        self.char = char
        self.cost = cost

class Autocomplete():
    def __init__(self, char=None, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node(char, node, 1)
                else:
                    node.children[char].cost += 1
                node = node.children[char]
            node.is_word = True
                

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        frontier = []
        suggestions = []
        currWord = ""
        node = self.root
        letters = []

        for i in range(len(prefix)):
            if prefix[i] in node.children:
                node = node.children[prefix[i]]
            else:
                node = self.root
                break
        
        if not (node == self.root):
            frontier.append(node)

        while frontier:
            currLetter = frontier.pop(0)
            for char in currLetter.children.keys():
                frontier.append(currLetter.children[char])


            if currLetter.is_word:
                while currLetter != self.root:
                    letters.append(currLetter.char)
                    currLetter = currLetter.parent
                letters = reversed(letters)
                for letter in letters:
                    currWord += letter
                suggestions.append(currWord)
                currWord = ""
                letters = []

        print(suggestions)
        return suggestions

            

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        frontier = []
        suggestions = []
        currWord = ""
        node = self.root
        letters = []

        for i in range(len(prefix)):
            if prefix[i] in node.children:
                node = node.children[prefix[i]]
            else:
                node = self.root
                break
        
        if not (node == self.root):
            frontier.append(node)

        frontier = deque(frontier)
        while frontier:
            currLetter = frontier.pop()
            while currLetter.children:
                for char in currLetter.children.keys():
                    frontier.append(currLetter.children[char])
                if currLetter.is_word:
                    while currLetter != self.root:
                        letters.append(currLetter.char)
                        currLetter = currLetter.parent
                    letters = reversed(letters)
                    for letter in letters:
                        currWord += letter
                    suggestions.append(currWord)
                    currWord = ""
                    letters = []
                currLetter = frontier.pop()


            if currLetter.is_word:
                while currLetter != self.root:
                    letters.append(currLetter.char)
                    currLetter = currLetter.parent
                letters = reversed(letters)
                for letter in letters:
                    currWord += letter
                suggestions.append(currWord)
                currWord = ""
                letters = []

        print(suggestions)
        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        frontier = []
        visited = []
        suggestions = []
        currWord = ""
        node = self.root
        letters = []
        pathCost = 0
        costs = []
        currCost = 0

        for i in range(len(prefix)):
            if prefix[i] in node.children:
                node = node.children[prefix[i]]
            else:
                node = self.root
                break
        
        if not (node == self.root):
            frontier.append(node)

        costs = []
        heapq.heapify(costs)

        while frontier:
            currLetter = frontier.pop()

            if currLetter in visited:
                continue

            visited.append(currLetter)

            for char in currLetter.children.keys():
                frontier.append(currLetter.children[char])
            
            for letter in frontier:
                if letter not in visited:
                    heapq.heappush(costs, 1/letter.cost)
            
            if (costs):
                currCost = heapq.heappop(costs)
                pathCost += currCost

            for letter in frontier:
                if letter not in visited:
                    if currCost == 1/letter.cost:
                        visited.append(letter)
                        for char in letter.children.keys():
                            frontier.append(letter.children[char])
                            letter.children[char].cost += 1/pathCost
                        if letter.is_word: 
                            while letter != self.root:
                                letters.append(letter.char)
                                letter = letter.parent
                            letters = reversed(letters)
                            for letter in letters:
                                currWord += letter
                            suggestions.append(currWord)
                            currWord = ""
                            letters = []


            if currLetter.is_word: 
                while currLetter != self.root:
                    letters.append(currLetter.char)
                    currLetter = currLetter.parent
                letters = reversed(letters)
                for letter in letters:
                    currWord += letter
                suggestions.append(currWord)
                currWord = ""
                letters = []
            
        print(suggestions)
        return suggestions
        


if __name__ == "__main__":
    from utilities import read_file
    autocomplete_engine = Autocomplete()
    filename = 'test.txt'
    read_file(filename, autocomplete_engine)
    curr = autocomplete_engine.root
    #print(curr.children)
    curr = curr.children['t']
    #print(curr.children)
    curr = curr.children['h']
    #print(curr.children)
    curr = curr.children['e']
    #print(curr.children)
    curr = curr.children['r']
    #print(curr.children)