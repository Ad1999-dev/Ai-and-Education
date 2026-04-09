from __future__ import annotations
from collections import deque
import heapq
import random
import string


class Node:
    def __init__(self):
        self.children: dict[str, Node] = {}
        self.freq_cost: float = 1
    
    def __lt__(self, other: Node):
        return self.freq_cost < other.freq_cost


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root: Node = Node()
        self.suggest = self.suggest_ucs

    
    def build_tree(self, document: str):
        for word in document.split():
            node: Node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                
                node = node.children[char]
            
            node.children['\0'] = Node()
        
        self._assign_freq_costs()
    
    
    # Depth-first-searches to assign frequency cost values
    def _assign_freq_costs(self):
        def assign_freq_costs_helper(node: Node) -> float:
            if not node.children:
                node.freq_cost = 0
                return 1
            
            num_leaves = 0
            for child in node.children.values():
                num_leaves += assign_freq_costs_helper(child)
            
            node.freq_cost = 1 / num_leaves
            
            return num_leaves
    
        assign_freq_costs_helper(self.root)


    def suggest_random(self, prefix: str) -> list[str]:
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]


    def suggest_bfs(self, prefix: str) -> list[str]:
        currNode: Node = self.root
        
        for char in prefix:
            currNode = currNode.children[char]
        
        deq: deque[tuple[Node, str]] = deque([(currNode, prefix)])
        suggestions: list[str] = []
        while deq:
            currNode, word = deq.popleft()
            
            for char, child in currNode.children.items():
                if char == '\0':
                    suggestions.append(word)
                    continue
                
                deq.append((child, word + char))
        
        return suggestions


    def suggest_dfs(self, prefix: str) -> list[str]:
        def suggest_dfs_helper(node: Node, word: str) -> list[str]:
            suggestions: list[str] = []
            
            for char, child in node.children.items():
                if char == '\0':
                    suggestions.append(word)
                    continue
                
                new_word = word + char
                suggestions += suggest_dfs_helper(child, new_word)
            
            return suggestions
        
        
        currNode: Node = self.root
        for char in prefix:
            currNode = currNode.children[char]
        
        return suggest_dfs_helper(currNode, prefix)


    def suggest_ucs(self, prefix: str) -> list[str]:
        currNode: Node = self.root
        
        for char in prefix:
            currNode = currNode.children[char]
        
        pq: list[tuple[float, Node, str]] = [(currNode.freq_cost, currNode, prefix)]
        suggestions: list[str] = []
        while pq:
            cost, currNode, word = heapq.heappop(pq)
            
            for char, child in currNode.children.items():
                if char == '\0':
                    suggestions.append(word)
                    continue
                
                heapq.heappush(pq, (cost + child.freq_cost, child, word + char))
        
        return suggestions
        
