# Reports section

## Late Days
How many late days are you using for this assigmment?
- Zero

## 383GPT
Did you use 383GPT at all for this assignment (yes/no)?
- Yes

## `build_tree`

### Tree diagram
- Tree diagram for `test.txt`:
<img width="333" alt="tree_diagram_test" src="https://github.com/user-attachments/assets/7e40fc22-6552-46f5-8d74-46c2d2b888b6" />

### Code analysis

- Raw:
```python
def build_tree(self, document):
        for word in document.split():
            node = self.root
            for char in word:
                if char not in node.children:
                    node.children[char] = Node()
                node = node.children[char]
            node.is_word = True
```

- Intuition:

The function loops through each word and attempts to insert it's characters into the tree one by one. It starts with the root node and if the character is not the child of that node, then it becomes a new node that is a child of that node. This continues until it marks the last character's node as the end of a valid word.

  

## `BFS`

### Code analysis

- Raw:
```python
def suggest_bfs(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        queue = deque([(node, prefix)])
        suggestions = []

        while queue:
            curr, word = queue.popleft()
            if curr.is_word:
                suggestions.append(word)

            for char, child in curr.children.items():
                queue.append((child, word + char))

        return suggestions
```
- Intuition:

It takes a prefix and searches until it finds a last character in the tree. If the characters in the prefix are not found in that order, it returns an empty list. Otherwise, it initializes a queue with the last character’s node and explores all possible word completions. Each time a node that completes the word is found, the word is added to the suggestions list. This process continues until the queue is empty, ensuring that all possible words starting with the prefix are explored.

### Your output

- Outputs after running with `suggest_bfs()` on the prefix "th" for `genZ.txt`:

the, thee, thou, that, thag, there, their, though, thought, through



## `DFS`

### Code analysis

- Raw:
```python
def suggest_dfs(self, prefix):
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        
        stack = [(node, prefix)]
        suggestions = []

        while stack:
            curr, word = stack.pop()
            if curr.is_word:
                suggestions.append(word)
            
            for char, child in reversed(curr.children.items()):
                stack.append((child, word + char))
        
        return suggestions
```

- Intuition:

It's similar to the previous function where it takes a prefix and searches for its last character in the tree, and if the characters in the prefix are not found in that order, it returns an empty list. Then it uses a stack with the last character’s node to explore all possible words. Each time a word completing node is found, it is added to the suggestions list. The function continues until the stack is empty, ensuring that all possible words starting with the prefix are found.

### Your output

- Outputs after running with `suggest_dfs()` on the prefix "th" for `genZ.txt`:

the, there, their, thee, thou, though, thought, that, thag, through

### Recursive DFS vs Stack-based DFS

- This method used stack-based DFS. 


## `UCS`

### Code analysis

- Raw:
```python
def suggest_ucs(self, prefix):

        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        priority_queue = [(0, node, prefix)]
        suggestions = []

        while priority_queue:
            cost, curr, word = heapq.heappop(priority_queue)
            if curr.is_word:
                suggestions.append(word)

            for char, child in curr.children.items():
                heapq.heappush(priority_queue, (cost + 1, child, word + char))

        return suggestions
```

- Intuition:

Like the other functions, it takes a prefix and searches for the last character in the tree and returns an empty list if they aren't found in that order. Otherwise it initialized a priority queue to loop through all word completions in order of increasing cost, where the cost is word length. Each time a word completing node is found it's added to suggestions and once all possible completing nodes with the starting prefixes are explored, it's returned.

### Your output

- Outputs after running with `suggest_ucs()` on the prefix "th" for `genZ.txt`:

N/A - no suggestions given

## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS.

BFS returned words in order of increasing length, so shorter words appeared first. DFS explored words by diving deeper first so it led to a slightly more alphabetical approach. UCS, however, did not return any words.







