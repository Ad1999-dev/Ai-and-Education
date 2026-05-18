from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.isEndOfTheWord = False # to check if it's the end of the word or not.
        self.frequencyOfChar = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                # check if the char is there or not. if not then add a node.
                if char not in node.children:
                    node.children[char] = Node()
                    node.frequencyOfChar[char] = 0 # setting the frequency of specific char to 0.
                node.frequencyOfChar[char] += 1 # incrementing the freq.
                node = node.children[char] # move to the next node.
            node.isEndOfTheWord = True # marking the end of the word
    ''' For each word in the document start from the root. And for each char in the word, check if it is 
        a child of the current node, if yes, increment the frequency of that char and move to the next node, 
        otherwise, create a new node and set the frequency of that char to 0.
        After the word is processed mark the last node as the end of the word.'''

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        # finding the node that matches the last char of the prefix.
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # Performing BFS from the node that got matched from the last char of the prefix.
        queue = deque([(node, prefix)])
        suggestions = []
        while queue:
            curr_node, curr_word = queue.popleft()
            if curr_node.isEndOfTheWord:
                suggestions.append(curr_word)

            for ch, child in curr_node.children.items():
                queue.append((child, curr_word + ch))
        return suggestions

    ''' In BFS we start from the last char of the given prefix. Then explore all the possible 
        words that can be formed by checking next char from the queue. If find a complete word at any node, 
        append it to the list of suggestions or else append the node to the queue while adding the curr_char to the 
        prefix.'''

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        # finding the node that matches the last char of the prefix.
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        # Performing DFS from the node that got matched from the last char of the prefix.
        # implementing stack based dfs.
        # stack = [(node, prefix)]
        suggestions = []
        # while stack:
        #     curr_node, curr_word = stack.pop()
        #     if curr_node.isEndOfTheWord:
        #         suggestions.append(curr_word)
        #     for ch, child in curr_node.children.items():
        #         stack.append((child, curr_word + ch))
        # return suggestions

        # defining helper function to use recursion.
        def dfs(curr_node, curr_word):
            if curr_node.isEndOfTheWord:
                suggestions.append(curr_word)
            for ch, child in curr_node.children.items():
                dfs(child, curr_word + ch)
        dfs(node, prefix)
        return suggestions
    
    ''' For implementing DFS I used recursive DFS. I have used a helper function which recursively calls 
        itself for each child node, diving deeper into the tree. Once it reaches the end of a branch, it 
        automatically backtracks and continues exploring other branches. And at each step, it checks if the 
        current node is the end of a word, if yes, the word is added to the suggestions list. This process 
        repeats until all possible words that start with the prefix have been found, and the complete list 
        of suggestions is returned.'''
    
    #TODO for students!!!
    def suggest_ucs(self, prefix):
        # finding the node that matches the last char of the prefix.
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        suggestions = []
        priority_queue = []

        heapq.heappush(priority_queue, (0, prefix, node))
        while priority_queue:
            total_cost, curr_word, curr_node = heapq.heappop(priority_queue)
            if curr_node.isEndOfTheWord:
                suggestions.append(curr_word)
            for ch, child in curr_node.children.items():
                freq = curr_node.frequencyOfChar[ch]
                cost = total_cost + (1/freq)
                heapq.heappush(priority_queue, (cost, curr_word + ch, child))
        return suggestions
    ''' For implementing UCS I did some changes in the function buildTree. I keep a track of the frequency 
        of characters to get the path cost.
        Then in the suggest_ucs, first i find the node that matches the last char of the prefix. Then, used
        a priority queue (min-heap) to perform UCS. The queue stores a tuple of (cost, word, curr_node).
        Then, while the queue is not empty:
                deque the node with lowest cost,
                check for word completion,
                    if it is the end of the word append it to suggestions 
                else, calculate the new cost and push the node to the queue.
                return the list.'''
