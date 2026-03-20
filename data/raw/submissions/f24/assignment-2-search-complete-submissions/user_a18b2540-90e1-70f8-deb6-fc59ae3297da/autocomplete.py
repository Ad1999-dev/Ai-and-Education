from collections import deque
import heapq
import random
import string


class Node:
    #TODO
    def __init__(self, data="", str=""):
        self.children = {}
        self.data = data
        self.str = str
        self.end = 0
        self.child_num = 0
        self.freq = 1
        self.pri = 0


class Autocomplete():
    def __init__(self, parent=None, document=""):
        self.root = Node()
        self.suggest = self.suggest_dfs #Default, change this to `suggest_dfs/ucs/bfs` based on which one you wish to use.

    
    
    def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                #TODO for students
                if char in node.children:
                    node = node.children[char]
                    node.freq = node.freq + 1
                else:
                    cur_str = node.str+char
                    node.children[char] = Node(char, cur_str)
                    node = node.children[char]
            node.end = 1
        node = self.root
        explored = []
        queue = []
        for child in node.children:
            queue.append(node.children[child])
        while queue:
            cur_node = queue.pop(0)
            cur_node.pri = 1/cur_node.freq
            if cur_node not in explored:
                explored.append(cur_node)
                for child in cur_node.children:
                    queue.append(cur_node.children[child])
        node = self.root

    def suggest_random(self, prefix):
        random_suffixes = [''.join(random.choice(string.ascii_lowercase) for _ in range(3)) for _ in range(5)]
        return [prefix + suffix for suffix in random_suffixes]

    #TODO for students!!!
    def suggest_bfs(self, prefix):
        pref_node = self.root
        for char in prefix:
            if char in pref_node.children:
                pref_node = pref_node.children[char]
            else:
                return []
        explored = []
        bfs_suffixes = []
        queue = []
        for child in pref_node.children.values():
            queue.append(child)
        while queue:
            cur_node = queue.pop(0)
            if cur_node.end == 1 and cur_node.str not in bfs_suffixes:
                bfs_suffixes.append(cur_node.str)
            if cur_node not in explored:
                explored.append(cur_node)
                for child in cur_node.children.values():
                    queue.append(child)
        return [suffix for suffix in bfs_suffixes]
    

    #TODO for students!!!
    def suggest_dfs(self, prefix):
        pref_node = self.root
        for char in prefix:
            if char in pref_node.children:
                pref_node = pref_node.children[char]
            else:
                return []
        explored = []
        dfs_suffixes = []
        stack = []
        for child in pref_node.children.values():
            stack.append(child)
        while stack:
            cur_node = stack.pop()
            if cur_node.end == 1 and cur_node.str not in dfs_suffixes:
                dfs_suffixes.append(cur_node.str)
            if cur_node not in explored:
                explored.append(cur_node)
                for child in cur_node.children.values():
                    stack.append(child)
        return [suffix for suffix in dfs_suffixes]


    #TODO for students!!!
    def suggest_ucs(self, prefix):
        pref_node = self.root
        for char in prefix:
            if char in pref_node.children:
                pref_node = pref_node.children[char]
            else:
                return []
        explored = []
        ucs_suffixes = []
        queue = []
        pri_queue = []
        added = []
        new_queue = []
        for child in pref_node.children:
            pri_queue.append(pref_node.children[child].pri)
        heapq.heapify(pri_queue)
        for i in range(0, len(pri_queue)):
            for child in pref_node.children:
                if pref_node.children[child].pri == pri_queue[i] and pref_node.children[child] not in added:
                    new_queue.append(pref_node.children[child])
                    added.append(pref_node.children[child])
        queue = new_queue
        while queue:
            cur_node = queue.pop(0)
            if cur_node.end == 1 and cur_node.str not in ucs_suffixes:
                ucs_suffixes.append(cur_node.str)
            if cur_node not in explored:
                explored.append(cur_node)
                pri_queue = []
                added = []
                new_queue = []
                for child in cur_node.children:
                    pri_queue.append(cur_node.children[child].pri)
                for i in range(0, len(queue)):
                    pri_queue.append(queue[i].pri)
                heapq.heapify(pri_queue)
                for i in range(0, len(pri_queue)):
                    for child in cur_node.children:
                        if cur_node.children[child].pri == pri_queue[i] and cur_node.children[child] not in added:
                            new_queue.append(cur_node.children[child])
                            added.append(cur_node.children[child])
                    for j in range(0, len(queue)):
                        if queue[j].pri == pri_queue[i] and queue[j] not in added:
                            new_queue.append(queue[j])
                            added.append(queue[j])
                queue = new_queue
        return [suffix for suffix in ucs_suffixes]
