from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.is_word = False
        self.following_character_frequency_count = {}
        self.path_cost_corresponding_to_previous_characters = {}

class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_bfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.
    
    def Caculate_inverse_for_each_following_character(self,node2):
        for neighbor, frequencies in node2.following_character_frequency_count.items():
            node2.path_cost_corresponding_to_previous_characters[neighbor] = 1/frequencies
            self.Caculate_inverse_for_each_following_character(node2.children[neighbor])
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char not in node.children:
                    node.children[char] = Node()
                if(char in node.following_character_frequency_count):
                    node.following_character_frequency_count[char] += 1
                else:
                    node.following_character_frequency_count[char] = 1
                node = node.children[char]
            node.is_word = True
        node2 = self.root
        self.Caculate_inverse_for_each_following_character(node2)

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]
    
    def iterate_through_the_Tree(self,prefix,tempN):
        for c in prefix:
            if(c in tempN.children):
                tempN = tempN.children[c]
            else:
                return ""
        return tempN

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        if(len(prefix) == 0):
            return []
        tempN = self.root
        result = []
        tempN = self.iterate_through_the_Tree(prefix,tempN)
        if(type(tempN) != Node):
            return "No suggestion"
        qWord_Level = deque()
        qWord_Level.append((prefix,tempN))
        while(qWord_Level):
            currCharacter, currPointerToThatCharacter  = qWord_Level.popleft()
            if(len(currPointerToThatCharacter.children.keys()) == 0 or currPointerToThatCharacter.is_word):
                result.append(currCharacter)
            for neighbor,PointerToTheNextNode in currPointerToThatCharacter.children.items():
                qWord_Level.append((currCharacter + neighbor, PointerToTheNextNode))
        return result
    
    #TODO for students!!!
    def recursive_function_helper(self,tempN,result,prefix):
        if(len(tempN.children.keys()) == 0 or tempN.is_word):
            result.append(prefix)
        for neighbor, nextLevel in tempN.children.items():
            nextPrefix = prefix + neighbor
            self.recursive_function_helper(nextLevel,result,nextPrefix)

    def suggest_dfs(self, prefix):
        if(len(prefix) == 0):
            return []
        tempN = self.root
        result = []
        tempN = self.iterate_through_the_Tree(prefix,tempN)
        if(type(tempN) != Node):
            return "No suggestion"
        self.recursive_function_helper(tempN,result,prefix)
        return result

    #TODO for students!!!
    def suggest_ucs(self, prefix):
        if(len(prefix) == 0):
            return []
        tempN = self.root
        result = []
        tempN = self.iterate_through_the_Tree(prefix,tempN)
        intial_cost = 0
        tempHeapq = []
        if(type(tempN) != Node):
            return "No suggestion"
        tempHeapq.append((intial_cost,prefix,tempN))
        while(tempHeapq):
            path_cost_so_far, sequence_letter_so_far, current_level_node = heapq.heappop(tempHeapq)
            if(current_level_node.is_word):
                result.append(sequence_letter_so_far)
            for neighbors,next_level_node in current_level_node.children.items():
                next_cost = path_cost_so_far + current_level_node.path_cost_corresponding_to_previous_characters[neighbors]
                heapq.heappush(tempHeapq,(next_cost,sequence_letter_so_far + neighbors, next_level_node))
        return result
