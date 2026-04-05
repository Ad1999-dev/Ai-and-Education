[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/UBg156UM)
# Assignment 2 - SearchComplete

## Assignment Objectives

1) Learn how to implement search algorithms in python
2) Learn how search algorithms can be used in practical application
3) Learning the differences between BFS, DFS, and UCS via implementation
4) Analyze the differences between search algorithms by comparing outputs
5) Learning how to build a search tree from textual data
6) Build a basic autocomplete feature that suggests words as the user types, using different search strategies.
7) Analyze how each algorithm affects the order and quality of suggestions, and learn when to choose each one.

## Pre-Requisites

- **Basic Python:** Familiarity with Python syntax, data structures (lists, dictionaries, queues), and basic algorithms.
- **Search Algorithms:** Theoretical understanding of BFS, DFS, and UCS
- **Tree:** Prior knowledge of Tree data structures is helpful.
- **Data Structures:** High level understanding of Data Structures like Stacks, Queues, and Priority Queues is required.

## Overview
Imagine you're an intern at a cutting-edge tech company called "WordWizard." Your first task: upgrade their revolutionary messaging app, "ChatCast," to include a mind-blowing autocomplete feature. The goal is simple – as users type, the app magically suggests the words they might be looking for, making conversations faster and more fun!

But here's the twist: Your quirky, genius boss, Dr. Lexico, insists on using classic search algorithms to power this futuristic feature. "Forget fancy neural networks," she exclaims. "Let's prove that good old BFS, DFS, and UCS can still deliver the goods!"

So, you're handed a massive dictionary of Gen Z slang and challenged to build the autocomplete engine. Can you master the algorithms, construct a word-filled tree, and unleash the power of search to create an autocomplete experience that will make even the most texting-savvy teen say, "OMG, this is lit!"?

The future of "ChatCast" (and your internship) depends on it. Time to dive into the code and become a word-suggesting wizard! 

## Lab Description

1. **First step**
    - Clone the repo and run `main.py`
      ```bash
      python main.py
      ```
    - If you're on linux/mac and the former doesn't work for you
      ```bash
      python3 main.py
      ```
      
      
2.  **Explore the Starter Code:**
    - Review the provided `Autocomplete` class. It handles building the tree from a text document, setting up a basic user interface, and providing a framework for the `suggest` method.
3.  **Implement Search Algorithms:**
    - Your main task is to complete the `suggest` methods. These methods should take a prefix as input and return a list of word suggestions. 
    - You'll implement multiple versions of `suggest`:
        - `suggest_bfs`: Breadth-First Search
        - `suggest_dfs`: Depth-First Search
        - `suggest_ucs`: Uniform-Cost Search  


## Background: Autocomplete as a Search Problem

Alright! Let's give you some context before you get into the weeds of the starter code. 
Autocomplete might seem like some complicated magic, but at its core, it's just an application of search algorithms on a tree (that's how it's done in this assignment for your simplicity, but it's done very differently in real word). Let's break down how this works:

**The Search Space: A Tree of Characters**

To implement the autocomplete feature, you would build a tree of characters, which will be the search space for this search problem. 
In your starter code, you're given a `document` (a `txt` file) of several words. 
Imagine each word in your document is broken down into its individual letters. Now, picture these letters arranged in a single tree-like structure, for example look at the tree diagram below:


**Tree Diagram**

For example, let the document that is given to you be - 

```txt
air ball cat car card carpet carry cap cape
```


```mermaid
graph TD;
    ROOT-->A_air[A];
    A_air[A]-->I_air[I]
    I_air[I]-->R_air[R]


    ROOT-->B
    B-->A_ball[A]
    A_ball[A]-->L_ball1[L]
    L_ball1[L]-->L_ball2[L]

    ROOT-->C
    C-->A_cat[A]
    A_cat[A]-->T

    A_cat[A]-->R
    R-->D

    R-->P_carpet[P]
    P_carpet[P]-->E_carpet[E]
    E_carpet[E]-->T_carpet[T]

    R-->R_carry[R]
    R_carry[R]-->Y

    A_cat[A]-->P_cape[P]
    P_cape[P]-->E_cape[E]

```

Above is a diagram of the tree that is build from the example `document` given above. Note how the *tree* starts with a common `root` 

- This is what the search space for your search problem would look like. 
- You will traverse the *tree* starting from the last node of the prefix that the user enters to generate autocomplete suggestions. 

**The Search Problem**

When a user types a prefix (e.g., "ca"), the autocomplete feature needs to find all the words in the *tree* that start with that prefix. This translates to a search problem:

- **Initial state:** The node representing the last letter of the prefix ("a" in our example).
- **Action** - a transition between one letter to the next letter in the *tree*
- **Goal:** The end of the word(s) (that start with the given prefix) in the *tree*. <u>Note how there could be multiple goals in this problem.</u>
- **Path:** The sequence of characters from the root to a goal node represents a complete word.

**Search Algorithms**

We can employ various search algorithms to traverse this *tree* and find our goal nodes (complete words).

- **Breadth-First Search (BFS):**  Explores the *tree* level-by-level, ensuring we find the shortest words first. 
- **Depth-First Search (DFS):** Dives deep into the *tree*, potentially finding longer, less common words first.
- **Uniform-Cost Search (UCS):** Considers the frequency of each character transition to prioritize more likely words based on the prefix.

**Multiple Goals and Paths**

In autocomplete, we're not just looking for a single goal node. We want to find *all* the goal nodes (words) that follow from the prefix. Furthermore, we're interested in the entire path from the root to each goal node, as this path represents the complete suggested word.

**Your Task:**

Your task is to implement BFS, DFS, and UCS to traverse the *tree* and generate autocomplete suggestions. You'll see how different algorithms affect the order and type of words suggested, and understand the trade-offs involved in choosing one over the other.


## Starter Code
For the starter code you have been given 3 files - 
1. **`autocomplete.py`** - This is where all your code that you write will go.
2. **`main.py`** - This file is responsible to setting up and running the autocomplete feature. Modifying this file is optional. Feel free to use this file for debugging or playing around with the autocomplete feature.
3. **`utilities.py`** - This file contains the code to read the document provided and building the Graphical User Interface for the autocomplete feature. This file is not related to the core logic of the autocomplete feature. Please do not modify this file.

### `autocomplete.py`
- This file has a `Node` class defined for you - 
    - Each Node represents a single character within a word. The `Node class has 1 attribute - 
        1. `children` - This is a dictionary that stores - 
            - Keys - Characters that which follow the current character in a word.
            - Values - `Node` objects, representing the next character in the sequence. 
    **You might (most likely will) want the `Node` class keep track of more things depending on how you implement you `suggest` methods.**

- The file also has an `autocomplete` class defined for you - 
    - The Engine Behind the Suggestions
    - **Attributes**
        - `root`: A root node of the tree. The tree stores all the words of the document in a tree structure, where each `Node` is character.
    - **Methods**
        - `__init__(document="")`:
            - Initializes an empty tree (the `root` node).
            - If a `document` string is provided, it builds the tree from that document.
            - document is a space separated textfile, example below.
            - ```txt
              air ball cat car card carpet carry cap cape
              ``` 
        - `build_tree(document)` #TODO:
            - As the name of the function suggests, takes a text string `document` and builds a tree of words, where each `Node` is a character. 
            - The implementationn of this method has been left up to you.

## **Student Tasks:**
The main goal of the lab activity is for students to implement the `build_tree`, `suggest_bfs`, `suggest_ucs`, and `suggest_dfs` methods. 


### 0. TODO: Intuition of the code written
- For all code that you will write for this assignment (which is not a lot), you must provide a breif intuition (1-2 sentences) of the major control structures of your code in the reports section at the bottom of this readme.
- You are not being asked to write a story, keep it concise and precise (remember, 1-2 sentences, at most 3).

**Consider the `fizz-buzz` code given below:**

```python
def fizzbuzz(n):
    for i in range(1, n + 1):
        if i % 15 == 0:
            print("FizzBuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
            print(i)

```

**Now this is what you're explaination should (somewhat) look like -**

<u>Iterates through a range of numbers n printing that number unless the number is a multiple of 3 or 5 where instead "Fizz" or "Buzz" is printed respectively. "FizzBuzz" is printed if the number is a multiple of both 3 and 5.</u>





### 1. TODO: `build_tree(document)`

>[!NOTE]
>**TODO: Draw the tree diagram of test.txt given in the starter code**
    - Upload the image into your `readme` into the reports section in the end of this readme.


**What it does:**

- Takes a text `document` as input.
- Splits the document into individual words.
- Inserts each word into a tree (prefix tree) data structure.
- Each character of a word becomes a node in the tree.

**Your task:**

- Complete the `for` loop within the `build_tree` method.




### 2. TODO: `suggest_bfs(prefix)`

**What it does:**

- Implements the Breadth-First Search (BFS) algorithm on the tree.
- Takes a `prefix` (the letters the user has typed so far) as input.
- Finds all words in the tree that start with the `prefix`.

**Your task:**
- Start from the node that corresponds to the last character of the `prefix`.
- Using BFS traverse the sub tree and build a list of suggestions.
- **Run your code with the `genZ.txt` file and `suggest_bfs()` method that you just implemented with the prefix `"th"` and note the the autocompleted suggestions it generates in the *Reports Section* below. Make sure you note down the suggestions in the same order in which they are originally displayed on your screen.**

### 3. TODO: `suggest_dfs(prefix)`

**What it does:**

- Implements the Depth-First Search (DFS) algorithm on the tree.
- Takes a `prefix` as input.
- Finds all words in the tree that start with the `prefix`.

**Your task:**
- Start from the node that corresponds to the last character of the `prefix`.
- Using DFS traverse the sub tree and build a list of suggestions.
- **Explain your intuition in recursive DFS VS stack-based DFS, and which one you used. Write this in the section provided at the end of this readme.**
- **Run your code with the `genZ.txt` file and `suggest_dfs()` method that you just implemented with the prefix `"th"` and note the the autocompleted suggestions it generates in the *Reports Section* below. Make sure you note down the suggestions in the same order in which they are originally displayed on your screen.**

### 4. TODO: `suggest_ucs(prefix)`

**What it does:**

- Implements the Uniform Cost Search (UCS) algorithm on the tree.
- Takes a `prefix` as input.
- Finds all words in the tree that start with the `prefix`.
- Prioritizes suggestions based on the frequency of characters appearing after previous characters.

**Your task:**

- Update `build_tree()` to store the path cost. The path cost is the inverse frequencies of that letter/char following that prefix of characters.
    - Using the inverse of these frequencies creates a lower path cost for more frequent character sequences.    
- Start from the node that corresponds to the last character of the `prefix`.
- Using UCS traverse the sub tree and build a list of suggestions.
- **Run your code with the `genZ.txt` file and `suggest_ucs()` method that you just implemented with the prefix `"th"` and note the the autocompleted suggestions it generates in the *Reports Section* below. Make sure you note down the suggestions in the same order in which they are originally displayed on your screen.**

<br>

>[!NOTE]
>This is not optional
> Try experimenting with different approaches and compare the results! Try typing different prefixes in the GUI and observe how the suggested words change depending on which search algorithm you're using. This will help you gain a deeper understanding of their strengths and weaknesses.<br>
> **Note down these observations in the reports section provided at the end of this readme**



## What to Submit

1.  **Completed `autocomplete.py` file:**  Containing your implementations of the `build_tree`, `suggest_bfs`, `suggest_dfs`, and `suggest_ucs` methods.
2.  **Completed _Reports Section_ at the botton of the `readme.md` file:** Briefly explaining wherever necessary, and completing the required tasks in the *Reports Section*. 

## Rubric

| Criteria                        | Points (Example) |
| -------------------------------- | ----------- |
| Diagram and explaination for `build_tree` | 10% |
| Correctness of `build_tree`      | 10%         |
| Explaination of `build_tree`      | 10%         |
| Correctness of `suggest_bfs`     | 10%         |
| Explaination of `suggest_bfs`     | 10%         |
| Correctness of `suggest_dfs`     | 10%         |
| Explaination of `suggest_dfs`     | 10%         |
| Correctness of `suggest_ucs`     | 10%         |
| Explaination of `suggest_ucs`     | 10%         |
| Experimention                     | 10 %        |

<hr>
<br>
<br>



# A Reports section

## 383GPT
Did you use 383GPT at all for this assignment (yes/no)?

I used 383GPT to understand some Python commands like pass, and to understand how to add properties to the Node class. I also asked 383GPT some questions about Python conditionals, types, and libraries like deque.

## `build_tree`

### Tree diagram
- Put the tree diagram for `test.txt` here

![Tree diagram](https://github.com/user-attachments/assets/39da2b78-09d3-4e36-827d-bdf5f58c1bb5)



### Code analysis

- Put the intuition of your code here

This function runs a for loop on the list of words in the document. Taking each word individually, it sets the current node to the root node and initializes i to 0. Another for loop runs on the word, taking each character individually. i is the number of characters that have been processed by the inner for loop, and it increments by 1 every time the inner loop runs. Inside the inner loop, if the current node does not have char as a child, a new branch is created. node.visits is incremented, to record the number of words passing through the node. The node pointer is moved further down the branch. If we’re at the last character of the word, node.end is incremented by 1, to denote that a word ends at this node. At the end, i is incremented.

### Your output

- Put the output you got for the prefixes provided here


## `BFS`

### Code analysis

- Put the intuition of your code here

This function runs a breadth first search on the tree. result is the array of suggested words. start is the node corresponding to the last letter of the prefix, and the helper method find_start is used to find this node. If start is None, then an empty list is returned, since a word starting with that prefix does not exist in the tree. A queue is initialized, and every element of the queue is of the form [word, next node]. Every character and node pair in start.children is appended to the right end of the queue. A loop runs while the queue is non-empty. Within each iteration, the leftmost word-node pair [word, node] is removed. If node is None, or if node.end is greater than 0, this implies that a word just ended at the node, and this word is added to the result list. Every word-node pair in this node’s children is added to the right end of the queue. At the end, the queue is empty, meaning that all nodes starting from the initial prefix have been processed. The result list is returned.

### Your output

- Put the output you got for the prefixes provided here

<img width="94" alt="BFS Output" src="https://github.com/user-attachments/assets/b0885d95-85a4-4fd2-a365-b0a1b147b819">



## `DFS`

### Code analysis

- Put the intuition of your code here

This function runs a depth first search on the tree. result is the array of suggested words. start is the node corresponding to the last letter of the prefix, and it is found using the find_start function. If start is None, an empty result list is returned because this means that the prefix does not exist in any word in the tree. successors is the reversed list of items in start.children. The list is reversed to ensure that the left child is added last and removed first (LIFO). All the [word, next node] pairs in successors are added to the right end of the stack. A loop runs while the stack is non-empty. Within every iteration, the word-node pair on the right end of the stack is removed. If the node is None or node.end is greater than 0, this means that a word ends at this node, and hence it is added to the result list. successors is the reversed list of items in node.children. Every word-node pair in successors is added to the right end of the stack. This process continues till the stack is empty. At the end, the result list is returned.

### Your output

- Put the output you got for the prefixes provided here

<img width="94" alt="DFS Output" src="https://github.com/user-attachments/assets/07d2683b-224a-4589-bf9c-79023119a89f">


### Recursive DFS vs Stack-based DFS

- Explain your intuition in recursive DFS VS stack-based DFS, and which one you used here.

I used a stack for DFS, because a stack is slightly more efficient than recursion in terms of space complexity. While the space complexity of both stack-based DFS and recursive DFS is O(bm), where b is the maximum degree of the tree and m is the maximum depth, a recursive algorithm uses more memory, since it also has to store duplicate values of variables used in the recursive function. On the other hand, a stack only needs to store the word-node pairs. In addition, I reversed the order in which child nodes are added to the stack, to make sure that nodes are processed in a left to right order. Also, if the maximum depth of the tree is very large, recursion can lead to a stack overflow, while a stack can handle larger depths. 


## `UCS`

### Code analysis

- Put the intuition of your code here

This function runs a uniform cost search on the tree. result is the array of suggested words. start is the node corresponding to the last letter of the prefix, and it is found using the find_start function. If start is None, an empty result list is returned because this means that the prefix does not exist in any word in the tree. pq is the priority queue, which is initialized to []. pq is then heapified to convert it into a min-heap, which can also be used as a priority queue. The element with the lowest cost will always be positioned at the 0th index (leftmost end) of pq. Every element of pq is a tuple of the form: (cost, word, next node). A tuple is made for every child of the start node, and the cost is the inverse of cnt, which is the sum of the number of visits (number of words involving the node) and the number of words ending at the node. The tuple is then pushed onto the heap. A loop runs while the heap has elements in it. Within each iteration, the topmost tuple in the heap (lowest cost) is removed. If the nxt node is empty or has words ending there, the word is added to the result list. A tuple is made for every item in nxt.children. The cost is set to the inverse of the frequency of character transitions, and the tuple is pushed onto the heap. At the end, pq is empty, and the result is returned. 

### Your output

- Put the output you got for the prefixes provided here

<img width="94" alt="UCS Output" src="https://github.com/user-attachments/assets/63f11654-a9f6-4324-84b6-5bb546fd1d3f">




## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS.

BFS: The suggested words are in the ascending order of their lengths. This occurs because BFS uses a queue with a first-in-first-out structure, and hence nodes are processed in the order of their depth from the root node. Thus, shorter words are added to the result earlier.
DFS: The suggested words are grouped by their 3rd letter. This occurs because DFS fully travels down a path until the word ends, and then backtracks to earlier nodes to find other paths to traverse. Thus, after “th”, the “e” node is visited, and all words starting with “the” are processed first. The process then continues for the prefixes “tho”, “tha”, and “thr”. 
UCS: In UCS, the lowest-cost path is processed first, and the cost of a transition is: (1 / character transition frequency). The prefix “the” is processed first because the number of character transitions from “h” (start node) to “e” are 4, which is higher than any other transition from the start node. The prefixes are then processed in the order: “h” to “o” (3 transitions), “h” to “a” (2 transitions), and “h” to “r”, (1 transition).








