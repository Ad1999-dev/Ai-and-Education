from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self):
        self.children = {}
        self.weight = 0 # node frequency, default 0

class Autocomplete():
    def __init__(self, parent=None, document="", algo='random'):
        self.root = Node()
        if algo == 'bfs':
            self.suggest = self.suggest_bfs
        elif algo == 'dfs':
            self.suggest = self.suggest_dfs
        elif algo == 'ucs':
            self.suggest = self.suggest_ucs
        else:
            self.suggest = self.suggest_random


    '''
    Creates a node for the char if it doesn't already exist in the parent's children and append it to the children dict. Records the appearance frequency of the most frequent word. Switches to the corresponding node and continues traversing the word string. Adds an END node when reaches leaves.
    '''
    def build_tree(self, document):
        word_dict = {}
        for word in document.split():
            node = self.root
            if word not in word_dict:
                word_dict[word] = 0
            word_dict[word] = word_dict[word] + 1
            for char in word:
                if not char in node.children.keys():
                    node.children[char] = Node()
                node = node.children[char]
                node.weight = max(word_dict[word], node.weight)
            if 'END' not in node.children.keys():
                node.children['END'] = Node()
            node.children['END'].weight = node.children['END'].weight + 1
        

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    '''
    A queue-based BFS algorithm. 
    Finds the last node in the word tree, returning empty list if the node can't be found. Then does a BFS search, controlled by `queue_nodes` and `queue_paths`, storing next nodes to visit and paths respectively. Whenever an `"END"` key is reached, the path gets stored in `words`, which gets returned after search is done.

    '''
    def suggest_bfs(self, prefix):
        if len(prefix) == 0:
            return []
        
        #traverse to the last node in prefix
        cur = self.root
        for char in prefix:
            if char in cur.children:
                cur = cur.children[char]
            else:
                return []
        #bfs
        words = []
        keys = sorted(cur.children.keys())
        if 'END' in keys:
            words.append(prefix)
            keys.remove('END')
        queue_nodes = [cur.children[key] for key in keys]
        queue_paths = [prefix + key for key in keys]
        while len(queue_nodes) > 0:
            cur = queue_nodes.pop(0)
            keys = sorted(cur.children.keys())
            if 'END' in keys:
                if len(keys) > 1:
                    words.append(queue_paths[0])
                    keys.remove('END')
                else:
                    words.append(queue_paths.pop(0))
                    continue
            queue_nodes.extend([cur.children[key] for key in keys])
            new_paths = [queue_paths[0] + key for key in keys]
            queue_paths.pop(0)
            queue_paths.extend(new_paths)
        return words
    

    '''
    A stack-based DFS algorithm. 
    Finds the last node in the word tree, returning empty list if the node can't be found. Then does a DFS search, controlled by `stack_nodes` and `stack_paths`, storing next nodes to visit and paths respectively. Whenever an `"END"` key is reached, the path gets stored in `words`, which gets returned after search is done.
    '''
    def suggest_dfs(self, prefix):
        if len(prefix) == 0:
            return []
        
        #traverse to the last node in prefix
        cur = self.root
        for char in prefix:
            if char in cur.children:
                cur = cur.children[char]
            else:
                return []
            
        #dfs
        words = []
        keys = sorted(cur.children.keys(), reverse=True)
        if 'END' in keys:
            words.append(prefix)
            keys.remove('END')
        stack_nodes = [cur.children[key] for key in keys]
        stack_paths = [prefix + key for key in keys]
        
        while len(stack_nodes) > 0:
            cur = stack_nodes.pop()
            keys = sorted(cur.children.keys(), reverse=True)
            if 'END' in keys:
                if len(keys) > 1:
                    words.append(stack_paths[-1])
                    keys.remove('END')
                else:
                    words.append(stack_paths.pop())
                    continue
            stack_nodes.extend([cur.children[key] for key in keys])
            new_paths = [stack_paths[-1] + key for key in keys]
            stack_paths.pop()
            stack_paths.extend(new_paths)
        return words

    '''
    The `suggest_ucs` method suggests words that begin with a given prefix from a trie data structure, organizing the suggestions by their associated weights. It traverses the trie to find the node corresponding to the end of the prefix and uses depth-first search to explore all possible completions. The method returns a list of words sorted by weight, ensuring that the most relevant suggestions are prioritized.
    '''
    def suggest_ucs(self, prefix):
        if len(prefix) == 0:
            return []
        
        #traverse to the last node in prefix
        cur = self.root
        for char in prefix:
            if char in cur.children:
                cur = cur.children[char]
            else:
                return []
        
        #ucs
        words = []
        stack_nodes = [cur]
        stack_paths = [prefix]
        stack_weights = [cur.weight]
        
        while len(stack_nodes) > 0:
            cur = stack_nodes.pop()
            cur_children = cur.children.copy()
            keys = list(cur.children.keys())

            if 'END' in keys:
                if len(keys) > 1:
                    cur_weight = cur.children['END'].weight
                    
                    new_node = Node()
                    new_node.weight = cur_weight
                    new_node.children['END'] = cur.children['END']
                    
                    keys.remove('END')
                    cur_children.pop('END')
                    
                    stack_nodes.append(new_node)
                    stack_paths.append(stack_paths[-1])
                    stack_weights.append(new_node.weight)
                else:
                    words.append(stack_paths.pop())
                    stack_weights.pop()
                    continue

            stack_nodes.extend([cur_children[key] for key in keys])
            
            new_paths = [stack_paths[-1] + key for key in keys]
            stack_paths.pop()
            stack_paths.extend(new_paths)

            new_weights = [cur_children[key].weight for key in keys]
            stack_weights.pop()
            stack_weights.extend(new_weights)
            
            #sort all stacks by weights
            indexes = list(range(len(stack_weights)))
            indexes = sorted(indexes, key=lambda index: stack_weights[index])
            temp_nodes = [stack_nodes[i] for i in indexes]
            temp_paths = [stack_paths[i] for i in indexes]
            temp_weights = [stack_weights[i] for i in indexes]
            stack_nodes = temp_nodes
            stack_paths = temp_paths
            stack_weights = temp_weights
        return words
            
                 
    
            
            

            
            