# A Reports section

## 383GPT
Did you use 383GPT at all for this assignment (yes/no)?
- No

## `build_tree`

### Tree diagram
- Put the tree diagram for `test.txt` here

[tree diagram](./build_tree_diagram.png)

### Code analysis

- Iterates through each character in each word in the document, creating a node for each unique character and attaching it to the previous (attaching the first unique letter to the root) and adding an end node to the last character of each word. Then recursively traverses to the leaves of the tree and and back up to the root, increasing the inverse of the cost by one for each end node (possible word) in the subtree.

## `BFS`

### Code analysis

- Travel to last prefix character node in the tree, and check if each child is an end node. If so, travel back to the root, assemble the word, add it to the suggestions list, and pop the first item in the queue, otherwise enqueue it's children, pop the first item in the queue, and repeat until the queue is empty.

### Your output

- Output for th:
    - the
    - thee
    - thou
    - that
    - thag
    - there
    - their
    - though
    - thought
    - through


## `DFS`

### Code analysis

- Travel to last prefix character node in the tree, and for each of its children extend the suggestions list with the result of the helper function, which check

### Your output

- Output for th
    - there
    - the
    - their
    - thee
    - though
    - thought
    - thou
    - that
    - thag
    - through

### Recursive DFS vs Stack-based DFS
- Recursive DFS
    - For each child of the start node, recurse on each of their children until the goal node is discovered or all nodes are discovered.
- Stack-based DFS
    - Place each child of the start node onto the stack. If the node on top isn't the goal node, remove it from the stack, add it's children on top, and repeat until the goal node is discovered or all nodes are discovered.


## `UCS`

### Code analysis

- Travels to last prefix character node in the tree, and places its children in a min heap using the cost of each node.
Explore the nodes as they're popped out the min heap. If an end node is encountered, traverse back up the root, spell the word and add it to suggestions, otherwise add it's children to the heap

### Your output

- Output for th:
    - the
    - thee
    - thag
    - their
    - there
    - that
    - thou
    - though
    - thought
    - through



## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS. 

BFS find the shortest words to match the prefix first, DFS groups words with the same prefixes or partial prefix together, and UCS groups words with the sam prefix, but not with the same partial prefix. For example, in UCS when I type t, the and thee are next to each other and their and there are next to each other, but they're not in on contiguous group, unlike in DFS






