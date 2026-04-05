[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/-fOB9vwA)
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
Did you use 383GPT at all for this assignment (yes/no)? YES

## `build_tree`

### Tree diagram
- Put the tree diagram for `test.txt` here

```txt
there though that the their through thee thou thought thag
```

```mermaid
graph TD;
    ROOT-->T
    T-->H
    H-->A
    H-->E
    H-->O
    H-->R
    
    A-->G
    A-->T_a[T]
    
    E-->E_thee[E];
    E-->I_their[I];
    E-->R_there[R];
    
    I_their[I]-->R_their[R];
    
    R_there[R]-->E_there[E];
    
    O-->U
    U-->G_u[G]
    G_u[G]-->H_u[H]
    H_u[H]-->T_u[T]
    
    R-->R_r[R]
    R_r[R]-->O_r[O]
    O_r[O]-->U_r[U]
    U_r[U]-->G_r[G]
    G_r[G]-->H_r[H]
```

### Code analysis

- Put the intuition of your code here:
This method takes a string `document`, splits it into individual words, and inserts each word into a trie data structure. For each word, it iterates through its characters, creating a new node for each character if it does not already exist in the current node's children. Once all characters of the word have been processed, it marks the final node as the end of a valid word.

### Your output

- Put the output you got for the prefixes provided here: 
No output, no search algorithms have been implemented yet.

## `BFS`

### Code analysis

- Put the intuition of your code here:
The function performs a breadth-first search to find all possible word completions starting with a given prefix. It first navigates to the node corresponding to the prefix in the trie. If the prefix isn't found, returns an empty list. Otherwise, it uses a queue to explore all possible paths from that node level by level. At each node, if it marks a complete word, that word is added to suggestions. For each node explored, all its children are added to the queue with their corresponding prefixes. The process continues until the queue is empty, ensuring all possible completions are found in order of increasing length.

### Your output

- Put the output you got for the prefixes provided here

the
thee
thou
that
thag
there
their
though
thought
through

## `DFS`

### Code analysis

- Put the intuition of your code here:
The algorithm first navigates to the node corresponding to the prefix. Using a stack for exploration, it follows each possible path completely before backtracking to try other paths. When a node marking a complete word is found, it's added to the suggestions list. The children of each node are added to the stack in reverse sorted order, ensuring deeper paths are explored first and results maintain lexicographic ordering. Unlike BFS which explores level by level, DFS will find the longest possible completion along one path before moving to another path. For example, with prefix "ca" in a trie containing "cat", "catch", "cats", it might first find "catch" (exploring the deepest path), then backtrack to find "cats", and finally "cat".

### Your output

- Put the output you got for the prefixes provided here

thag
that
the
thee
their
there
thou
though
thought
through

### Recursive DFS vs Stack-based DFS
- Explain your intuition in recursive DFS VS stack-based DFS, and which one you used here.

The `suggest_dfs` method employs a stack-based Depth-First Search (DFS) approach to explore the Trie data structure. This method has several advantages:

Explicit Control: By using a stack, we have direct control over which nodes to explore. We push nodes onto the stack when we go deeper and pop them off when we backtrack. This makes the flow of exploration clear and manageable.
Avoids Recursion Limitations: Unlike recursive DFS, which relies on the call stack and can cause issues if the structure is too deep (potentially leading to a stack overflow error), the stack-based approach removes this limitation. We can handle larger data structures without worrying about hitting a recursion depth limit.
Maintains Order: The stack allows us to manage the order in which we process nodes. By carefully pushing and popping nodes, we ensure that we explore all possible paths and suggestions in a structured way.

Using stack-based DFS in `suggest_dfs` provides a reliable and efficient way to gather autocomplete suggestions from the Trie, while keeping the code straightforward and avoiding the pitfalls of recursion.

## `UCS`

### Code analysis

- Put the intuition of your code here
The `suggest_ucs` method utilizes the Uniform Cost Search (UCS) algorithm to generate word suggestions from a trie based on a given prefix. It starts by locating the node that corresponds to the end of the prefix.
The algorithm employs a priority queue to explore possible completions, where each node's cost is determined by the inverse frequency of the characters that follow the prefix (i.e., `1/frequency`). This means that more frequently occurring characters have a lower cost, making them more likely to be suggested first.
To ensure unique suggestions, a set keeps track of explored prefixes. The algorithm retrieves the lowest-cost paths from the queue, adds valid completions to the suggestions, and continues to explore child nodes until all possibilities are exhausted. The final list of suggestions is ordered by increasing cost, prioritizing the most relevant and likely character sequences for user input.

### Your output

- Put the output you got for the prefixes provided here

the
thou
though
thee
thag
that
thought
their
there
through

## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS. 

BFS (Breadth-First Search): explores nodes level by level, prioritizing paths based on their depth from the starting point. As BFS proceeds, it examines all possible completions at a given depth before moving on to longer completions. This means that the output will list word completions in order of their length, starting from the shortest completions. Furthermore, within the same depth level, BFS typically generates completions in lexicographical order (alphabetical order), meaning that if two completions have the same length, the one that comes first alphabetically will be listed first.
DFS (Depth-First Search): explores deeper paths first, which leads to suggestions that prioritize longer completions before shorter ones. As it traverses down each branch of the tree, DFS will fully explore each path until it reaches a leaf node (a complete word) before backtracking. As a result, the suggestions generated will start with the longest completions and will generally be ordered alphabetically within that length. This means that the longest words may be suggested before any shorter alternatives, which can be less useful for users looking for quick, shorter completions.
UCS (Uniform Cost Search): prioritizes nodes based on the cumulative cost to reach them, where the cost in this context is determined by the inverse frequencies of characters following the prefix (rather than simply the number of characters added to the prefix). UCS uses a priority queue to ensure that the paths with the lowest cost are explored first. This approach allows UCS to provide suggestions that favor more common character sequences, aligning with typical user input patterns. As a result, UCS generates suggestions that are not only valid completions of the prefix but also those most likely to be typed next by users, which are typically shorter or more frequent completions. This means that, unlike BFS and DFS, UCS can suggest completions that may be deeper in the trie but more relevant based on frequency, often providing a more intuitive user experience.
Summary of Differences:
BFS suggests completions level by level based on length, typically in alphabetical order.
DFS prioritizes longer completions and explores deeper paths first, giving precedence to longer words, also in alphabetical order.
UCS, however, utilizes a cost-based approach to prioritize suggestions, favoring more frequently encountered character sequences, which can lead to quicker and more user-oriented completions, irrespective of depth.




