from collections import deque
import heapq
import random
import string
import sys


class Node:
    #TODO
    def __init__(self):
        self.is_last_letter = False
        self.letter = ''
        self.times_seen = 0
        self.transition_cost = 0
        self.path_cost = 0
        self.previous_letter = None
        self.children = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_ucs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    def build_tree(self, document):
        totalWords = len(document.split())

        for word in document.split():
            word_length = len(word)
            letter_counter = 0
            node = self.root
            for char in word:
                letter_counter += 1
                #TODO for students
                if (char in node.children):
                    child = node.children.get(char)
                    child.times_seen += 1
                    child.transition_cost = totalWords / child.times_seen
                    node = node.children.get(char)
                    
                    if letter_counter == word_length:
                        node.is_last_letter = True
                else:
                    child = Node()
                    child.previous_letter = node
                    child.char = char
                    node.children[char] = child
                    child.times_seen = 1
                    child.transition_cost = totalWords 
                    node = child

                    if letter_counter == word_length:
                        node.is_last_letter = True
                           

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        suggestions = []
        pointer = self.root
        queue = {}

        # Advances pointer to the last character of the prefix
        for i in range(len(prefix)):
            if prefix[i] in pointer.children:
                pointer = pointer.children.get(prefix[i])

            # If the prefix isn't part of any word, return an empty suggestion
            else:
                return {}

        # If the prefix a word, it will return it back
        if pointer.is_last_letter:
            suggestions.append(prefix)
        
        # If the prefix has any possible letters after it, add the prefix + the letter to the queue 
        for char in pointer.children.keys():
            queue[pointer.children[char]] = prefix + char

        # Iterates through the queue to find the shortest words first
        while len(queue) > 0:

            # Looks at the first entry in the queue
            char_node = list(queue)[0]

            # Adds the word if it is part of a longer word
            if char_node.is_last_letter and char_node.children:
                suggestions.append(queue[char_node])

            # If no letters follow, then it is a complete word. Add it to the suggestions
            if not char_node.children:
                suggestions.append(queue[char_node])
                queue.pop(char_node)

            # If there are more possible letters, add them to the existing string of letters and 
            # append them to the queue
            else:
                for next_letter in char_node.children.keys():
                    queue[char_node.children[next_letter]] = queue[char_node] + next_letter
                queue.pop(char_node)
                    
        return suggestions


    #TODO for students!!!
    def suggest_dfs(self, prefix):
        suggestions = []
        pointer = self.root
        queue = {}

        # Advances pointer to the last character of the prefix
        for i in range(len(prefix)):
            if prefix[i] in pointer.children:
                pointer = pointer.children.get(prefix[i])

            # If the prefix isn't part of any word, return an empty suggestion
            else:
                return {}
        
        # If the prefix has any possible letters after it, add the prefix + the letter to the queue 
        for char in pointer.children.keys():
            queue[pointer.children[char]] = prefix + char
            
        # If the prefix a word, it will return it back
        if pointer.is_last_letter:
            suggestions.append(prefix)
        
        while len(queue) > 0:

            # Looks at the last entry to the stack
            queue_list = list(queue)
            char_node = queue_list[len(queue_list) - 1]

            # Adds the word if it is part of a longer word
            if char_node.is_last_letter and char_node.children:
                suggestions.append(queue[char_node])    

            # If no letters follow, then it is a complete word. Add it to the suggestions
            if not char_node.children:
                suggestions.append(queue[char_node])
                queue.pop(char_node)

            # If there are more possible letters, add them to the existing string of letters and 
            # append them to the queue
            else:
                for next_letter in char_node.children.keys():
                    queue[char_node.children[next_letter]] = queue[char_node] + next_letter
                queue.pop(char_node)      

        return suggestions


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        suggestions = []
        pointer = self.root
        queue = {}
        current_path_cost = 0

        # Advances pointer to the last character of the prefix
        for i in range(len(prefix)):
            if prefix[i] in pointer.children:
                current_path_cost += pointer.path_cost
                pointer = pointer.children.get(prefix[i])
                pointer.path_cost += pointer.transition_cost + current_path_cost

            # If the prefix isn't part of any word, return an empty suggestion
            else:
                return {}
        
        # If the prefix has any possible letters after it, add the prefix + the letter to the queue 
        for char in pointer.children.keys():
            queue[pointer.children[char]] = prefix + char
            
        while len(queue) > 0:     

            # Calculates the next path cost
            nextChar = list(queue)[0]
            nextChar.path_cost = nextChar.previous_letter.path_cost + nextChar.transition_cost

            # Compares the next path cost to the other letters in the queue.
            # If there is a more cost efficient path, use that as the next letter
            for letter in list(queue):
                if letter.path_cost == 0:
                    letter.path_cost = letter.transition_cost + letter.previous_letter.path_cost
                if letter.path_cost < nextChar.path_cost:
                    nextChar = letter

            # Adds the word if it is part of a longer word
            if nextChar.is_last_letter and nextChar.children:
                suggestions.append(queue[nextChar])

            # If no letters follow, then it is a complete word. Add it to the suggestions
            if not nextChar.children:
                suggestions.append(queue[nextChar])
                queue.pop(nextChar)
            
            # If there are more possible letters, add them to the existing string of letters and 
            # append them to the queue
            else:
                for next_letter in nextChar.children.keys():
                    queue[nextChar.children[next_letter]] = queue[nextChar] + next_letter
                queue.pop(nextChar)
            
        return suggestions
