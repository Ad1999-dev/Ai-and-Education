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
<br />
Yes, but I did not use the code directly.

## `build_tree`

### Tree diagram
- Put the tree diagram for `test.txt` here
![tree](https://github.com/user-attachments/assets/46b7df05-1424-4204-a4e3-03e01ca9dd6b)

### Code analysis

- Put the intuition of your code here
<br />
I first modified the node class to make the build tree function work properly. I added a few parameters to the class, the final attribute (boolean) and the frequency attribute (int) to aid in searches and UCS.
<br />
<br />
As for build_tree, I made it so that it traverses each word character by character, starting from the root node. Each node represents a letter and has a dictionary of letters containing any/all of the subsequent letters that follow. So build tree traverses starting from the root node, and trying to find each subsequent letter, creating a new node where a letter does not exist, or just finding the reference to the existing node when one does exist. Upon encountering each node, it decrements its frequency by 1 (we are storing inverse frequencies for UCS) so that letters in a certain order that appear more frequently has a lower (and thus higher priority) frequency value in the node. Finally, once it reaches the last node in the word, it marks that node's final attribute to be true, since a valid word can end there.

### Your output

- Put the output you got for the prefixes provided here 
<br />
N/A

## `BFS`

### Code analysis

- Put the intuition of your code here
<br />
All of the search algorithms utilize an internal helper function called traverse_prefix. This function takes in a prefix and directly explores that path, returning the node of the last letter in the prefix. BFS begins by calling the helper function and storing that end-of-prefix node as "prefix_end."
<br />
<br />
BFS works such that we use a queue for our frontier. Each element of the queue should keep track of two pieces of information; the node reference and the string path up to and including that node. We will store both pieces of information in a tuple, (node reference, prefix string). The queue is initialized with the prefix end node and the string so far (starting as the prefix). We pop from the queue, adding the prefix if we reach a node where a word ends, and enqueue-ing the neighbors nodes and the prefix + the letter of the neighbor. We continue to do this until the queue is empty.


### Your output

- Put the output you got for the prefixes provided here <br />
<img width="191" alt="bfs" src="https://github.com/user-attachments/assets/e4a48457-508e-4a04-a5a7-e85a0e75e09c">



## `DFS`

### Code analysis

- Put the intuition of your code here

<br />

The __traverse_prefix(prefix) call tries to locate the node in the trie that represents the last character of the prefix. This node serves as the starting point for the DFS. If no such node exists (i.e., the prefix does not exist in the trie), it returns an empty list.

<br />
<br />

Starting from the prefix_end node, the DFS function traverses all possible child nodes recursively, searching for valid words by exploring each path from the prefix_end. At each node, it checks if the cur.final is True, meaning the node marks the end of a valid word. If it does, the word formed by concatenating the prefix and suffix (which contains characters traversed after the prefix) is added to the suggestions list. The function then recursively explores each child node, passing along the accumulated suffix (the characters traversed after the prefix), extending the suffix as it goes deeper.


### Your output

- Put the output you got for the prefixes provided here
<br />
<img width="188" alt="dfs" src="https://github.com/user-attachments/assets/1a122fc2-7041-48e1-a507-3b3726da6d58">


### Recursive DFS vs Stack-based DFS
- Explain your intuition in recursive DFS VS stack-based DFS, and which one you used here.

<br />
Recursive DFS utilizes a call stack while iterative stack based uses an explicit stack/
<br />
<br />

I chose to do the recursive over iterative approach for clean code and overall code readability. With recursion, you don't need to explicitly manage the stack or handle backtracking; the recursive call stack automatically does that for you. In an iterative solution, you would need to use a stack data structure to store not only the nodes but also the corresponding states, like the current suffix being built.

## `UCS`

### Code analysis

- Put the intuition of your code here
<br />
UCS is an algorithm that utilizes a priority queue as its frontier, prioritizing shortest path nodes first. In our case, we want to prioritize the nodes which appear after the previously seen letters most frequently, so we will keep track of frequency as an inverse count for edges such that nodes with a higher frequency will be smaller in value and move to the front of the queue with highest priority. We initialize their frequency by incrementing it for each edge when we build the tree. Every time we traverse an edge while adding a word to the tree, we increment the frequency by one. Again, this frequency will later be evaluated as an inverse.
<br />
<br />
We are using a priority queue (min-heap) to ensure the least "costly" paths are explored first. Each node in the trie represents a character with an associated frequency, and UCS navigates through the trie by expanding the most frequent characters, effectively suggesting the most common words first. The cost function is inversely proportional to the character frequency (1/freq), so more frequent characters incur lower costs, allowing the UCS to prioritize common word completions. Each node that is enqueued is enqueued with the value of the path up to that point, where the path value is equal to the sum of each edge up to that node.
<br />
<br />
The pq (priority queue) is initialized with a tuple: (0, 0, prefix_end, prefix) where: the first 0 represents the initial cost (or path) starting at zero, the second 0 is a counter to keep track of the order in which nodes enter the priority queue (this counter is used to break ties), and prefix_end is the node representing the last character of the prefix in the trie, and prefix is the current word or prefix.
<br />
<br />
This means we are starting from the node corresponding to the prefix and will explore the children nodes to find valid completions.
<br />
<br />
So to summarize, UCS utilizes a priority queue which prioritizes shortest path cost where path length is inversely proportional to frequency. We keep track of the length of the collective path traversed so far, and we break ties with order in which they entered the queue.
### Your output

- Put the output you got for the prefixes provided here
<br />

<img width="204" alt="Screenshot 2024-09-25 at 10 51 47 AM" src="https://github.com/user-attachments/assets/dfe0dd05-3029-41b3-a65a-4c7c474ef243">





## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS. 
<br />

I played around a lot with the different algorithms. I will compare my findings. First, when working with BFS, I noticed that it prioritized the shorter words first, then generated future suggestions by increasing word length. This might be useful when you want to prioritize shorter words, but in reality I'm not sure how practical it is.
<br />
<br />
Next, I observed DFS. DFS explores each branch deeply before moving on to the next branch. This means that suggestions may not be in order of word length or frequency. Depending on the trie structure, it can return longer words first. DFS can be useful when exhaustive search along one particular branch is desired. However, it does not guarantee an ordered list of results based on word length or frequency, making it less predictable for autocomplete systems.
<br />
<br />
Finally, I checked out UCS. This was my favorite and the one I thought made the most sense. UCS prioritizes suggestions based on character frequency. More frequent words (i.e., those with common characters) are returned first, regardless of word length. UCS ensures that the first few suggestions are typically the most common and useful completions for the prefix. UCS is ideal for generating the most probable word completions based on corpus data, making it highly effective in real-world autocompletion where frequency matters.