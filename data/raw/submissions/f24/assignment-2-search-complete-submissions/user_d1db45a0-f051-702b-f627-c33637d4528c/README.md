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

## `build_tree`

### Tree diagram
- Put the tree diagram for `test.txt` here

```mermaid
graph TD;
4315090240[ROOT]-->4315164992[A]
4315090240[ROOT]-->4315155328[B]
4315090240[ROOT]-->4315090752[C]
4315090240[ROOT]-->4315165504[D]
4315090240[ROOT]-->4315159488[E]
4315090240[ROOT]-->4315156224[F]
4315090240[ROOT]-->4315162560[G]
4315090240[ROOT]-->4315161856[H]
4315090240[ROOT]-->4315164032[I]
4315090240[ROOT]-->4311527808[L]
4315090240[ROOT]-->4315165248[M]
4315090240[ROOT]-->4315090624[N]
4315090240[ROOT]-->4315163648[O]
4315090240[ROOT]-->4315166336[P]
4315090240[ROOT]-->4315165760[Q]
4315090240[ROOT]-->4315164480[R]
4315090240[ROOT]-->4315159808[S]
4315090240[ROOT]-->4315157248[T]
4315090240[ROOT]-->4315165120[U]
4315090240[ROOT]-->4315162304[V]
4315090240[ROOT]-->4315166080[W]
4315164992[A]-->4315165056[F]
4315155328[B]-->4315155968[A]
4315155328[B]-->4315155392[E]
4315155328[B]-->4315155776[R]
4315155968[A]-->4315156032[S]
4315156032[S]-->4315156096[I]
4315156096[I]-->4315156160[C]
4315155392[E]-->4315155520[S]
4315155392[E]-->4315155456[T]
4315155520[S]-->4315155584[T]
4315155584[T]-->4315155648[I]
4315155648[I]-->4315155712[E]
4315155776[R]-->4315155840[U]
4315155840[U]-->4315155904[H]
4315090752[C]-->4315090816[A]
4315090752[C]-->4315091392[H]
4315090752[C]-->4315155072[L]
4315090752[C]-->4315154816[U]
4315090816[A]-->4315090944[N]
4315090816[A]-->4315090880[P]
4315090944[N]-->4315091008[C]
4315091008[C]-->4315091072[E]
4315091072[E]-->4315091136[L]
4315091136[L]-->4315091200[L]
4315091200[L]-->4315091264[E]
4315091264[E]-->4315091328[D]
4315091392[H]-->4315091520[E]
4315091520[E]-->4315154688[C]
4315154688[C]-->4315154752[K]
4315155072[L]-->4315155136[O]
4315155136[O]-->4315155200[U]
4315155200[U]-->4315155264[T]
4315154816[U]-->4315154880[R]
4315154880[R]-->4315154944[V]
4315154944[V]-->4315155008[E]
4315165504[D]-->4315165568[R]
4315165568[R]-->4315165632[I]
4315165632[I]-->4315165696[P]
4315159488[E]-->4315159552[X]
4315159552[X]-->4315159616[T]
4315159616[T]-->4315159680[R]
4315159680[R]-->4315159744[A]
4315156224[F]-->4315156288[A]
4315156224[F]-->4315156608[I]
4315156224[F]-->4315157056[L]
4315156224[F]-->4315156992[R]
4315156288[A]-->4315156416[C]
4315156288[A]-->4315156352[M]
4315156416[C]-->4315156480[T]
4315156480[T]-->4315156544[S]
4315156608[I]-->4315156800[N]
4315156608[I]-->4315156672[R]
4315156800[N]-->4315156864[N]
4315156864[N]-->4315156928[A]
4315156672[R]-->4315156736[E]
4315157056[L]-->4315157120[E]
4315157120[E]-->4315157184[X]
4315162560[G]-->4315162624[H]
4315162560[G]-->4315163072[L]
4315162560[G]-->4315163392[O]
4315162624[H]-->4315162688[O]
4315162688[O]-->4315162752[S]
4315162752[S]-->4315162816[T]
4315162816[T]-->4315162880[I]
4315162880[I]-->4315162944[N]
4315162944[N]-->4315163008[G]
4315163072[L]-->4315163136[O]
4315163136[O]-->4315163200[W]
4315163200[W]-->4315163264[E]
4315163264[E]-->4315163328[D]
4315163392[O]-->4315163456[A]
4315163456[A]-->4315163520[L]
4315163520[L]-->4315163584[S]
4315161856[H]-->4315161920[I]
4315161920[I]-->4315161984[G]
4315161984[G]-->4315162048[H]
4315162048[H]-->4315162112[K]
4315162112[K]-->4315162176[E]
4315162176[E]-->4315162240[Y]
4315164032[I]-->4315164160[C]
4315164032[I]-->4315164096[S]
4315164160[C]-->4315164224[O]
4315164224[O]-->4315164288[N]
4315164288[N]-->4315164352[I]
4315164352[I]-->4315164416[C]
4311527808[L]-->4313759424[I]
4311527808[L]-->4315090304[O]
4313759424[I]-->4312268032[T]
4312268032[T]-->4313759232[E]
4313759232[E]-->4313753472[R]
4315090304[O]-->4315090368[W]
4315090368[W]-->4315090432[K]
4315090432[K]-->4315090496[E]
4315090496[E]-->4315090560[Y]
4315165248[M]-->4315165312[O]
4315165312[O]-->4315165376[O]
4315165376[O]-->4315165440[D]
4315090624[N]-->4315090688[O]
4315163648[O]-->4315163712[U]
4315163712[U]-->4315163776[T]
4315163776[T]-->4315163840[F]
4315163840[F]-->4315163904[I]
4315163904[I]-->4315163968[T]
4315166336[P]-->4315166400[E]
4315166400[E]-->4315166464[R]
4315166464[R]-->4315166528[I]
4315166528[I]-->4315166592[O]
4315166592[O]-->4315166656[D]
4315166656[D]-->4315166720[T]
4315165760[Q]-->4315165824[U]
4315165824[U]-->4315165888[E]
4315165888[E]-->4315165952[E]
4315165952[E]-->4315166016[N]
4315164480[R]-->4315164544[E]
4315164544[E]-->4315164608[C]
4315164608[C]-->4315164672[E]
4315164672[E]-->4315164736[I]
4315164736[I]-->4315164800[P]
4315164800[P]-->4315164864[T]
4315164864[T]-->4315164928[S]
4315159808[S]-->4315159872[A]
4315159808[S]-->4315160384[H]
4315159808[S]-->4315160768[I]
4315159808[S]-->4315161664[L]
4315159808[S]-->4315161216[N]
4315159808[S]-->4315160640[U]
4315159872[A]-->4315159936[L]
4315159872[A]-->4315160128[V]
4315159936[L]-->4315160000[T]
4315160000[T]-->4315160064[Y]
4315160128[V]-->4315160192[A]
4315160192[A]-->4315160256[G]
4315160256[G]-->4315160320[E]
4315160384[H]-->4315160448[O]
4315160448[O]-->4315160512[O]
4315160512[O]-->4315160576[K]
4315160768[I]-->4315160832[M]
4315160768[I]-->4315161152[S]
4315160832[M]-->4315160896[P]
4315160896[P]-->4315160960[I]
4315160960[I]-->4315161024[N]
4315161024[N]-->4315161088[G]
4315161664[L]-->4315161728[A]
4315161728[A]-->4315161792[Y]
4315161216[N]-->4315161280[A]
4315161280[A]-->4315161344[T]
4315161344[T]-->4315161408[C]
4315161408[C]-->4315161472[H]
4315161472[H]-->4315161536[E]
4315161536[E]-->4315161600[D]
4315160640[U]-->4315160704[S]
4315157248[T]-->4315157312[B]
4315157248[T]-->4315157440[E]
4315157248[T]-->4315158208[H]
4315157248[T]-->4315157568[O]
4315157248[T]-->4315157952[U]
4315157312[B]-->4315157376[H]
4315157440[E]-->4315157504[A]
4315158208[H]-->4315158976[A]
4315158208[H]-->4315158272[E]
4315158208[H]-->4315158656[O]
4315158208[H]-->4315159168[R]
4315158976[A]-->4315159104[G]
4315158976[A]-->4315159040[T]
4315158272[E]-->4315158592[E]
4315158272[E]-->4315158464[I]
4315158272[E]-->4315158336[R]
4315158464[I]-->4315158528[R]
4315158336[R]-->4315158400[E]
4315158656[O]-->4315158720[U]
4315158720[U]-->4315158784[G]
4315158784[G]-->4315158848[H]
4315158848[H]-->4315158912[T]
4315159168[R]-->4315159232[O]
4315159232[O]-->4315159296[U]
4315159296[U]-->4315159360[G]
4315159360[G]-->4315159424[H]
4315157568[O]-->4315157632[T]
4315157632[T]-->4315157696[A]
4315157696[A]-->4315157760[L]
4315157760[L]-->4315157824[L]
4315157824[L]-->4315157888[Y]
4315157952[U]-->4315158016[R]
4315158016[R]-->4315158080[N]
4315158080[N]-->4315158144[T]
4315165120[U]-->4315165184[P]
4315162304[V]-->4315162368[I]
4315162368[I]-->4315162432[B]
4315162432[B]-->4315162496[E]
4315166080[W]-->4315166144[O]
4315166144[O]-->4315166208[K]
4315166208[K]-->4315166272[E]
```

### Code analysis

- Put the intuition of your code here

For each word in the list, the algorithm starts at the root node and for each character either creates a node with a key for that character (or does nothing if the key already exists) before going into that node and repeating the process at the next character. At the end of the word, the algorithm sets the "word" property for the leaf node to a string holding the word.

## `BFS`

### Code analysis

- Put the intuition of your code here

A queue holds a subset of the nodes of the tree, starting with only the root node. At each step, the oldest node on the queue gets taken off and its children are added in alphabetical order. When a node representing the end of a word is found, the string at this node (one is stored in every such node) gets added to the output list. This continues until the queue is empty.

### Your output

- Put the output you got for the prefixes provided here

the, thag, that, thee, thou, their, there, though, thought, through


## `DFS`

### Code analysis

- Put the intuition of your code here

A stack holds a subset of the nodes of the tree, starting with only the root node. At each step, the highest node on the stack gets get taken off and its children are added in reverse alphabetical order (so that the lowest in the alphabet ends up at the top of the stack). When a node representing the end of a word is found, the string at this node (one is stored in every such node) gets added to the output list. This continues until the queue is empty.

### Your output

- Put the output you got for the prefixes provided here

thag, that, the, thee, their, there, thou, though, thought, through

### Recursive DFS vs Stack-based DFS
- Explain your intuition in recursive DFS VS stack-based DFS, and which one you used here.

If you're just asking me what each one is, I described stack-based DFS above since that's the implementation I chose. Meanwhile, recursive DFS would function by doing a recursive call on each child node of the node you are on, which would effectively lead to a search on each next-layer subtree of the tree you are currently looking at. The recursive nature of this implementation makes it depth-first since you will only start going back up the call stack once a leaf node has been found. I chose to use the stack implementation however since it's more similar to BFS, meaning I could make a helper function which does either depending on a parameter which tells it whether it should use a stack or a queue.

## `UCS`

### Code analysis

- Put the intuition of your code here

A heap holds a subset of the nodes of the tree, starting with only the root node. Each node is held in the heap in a triple where the first value is its total distance from the root, the second value is the memory id of the node (just as a tiebreaker), and the third value is the node itself. At each step, the lowest-distance node on the heap gets taken off and its children are added in alphabetical order along with their distance from their parent added to the total distance of their parent. When a node representing the end of a word is found, the string at this node (one is stored in every such node) gets added to the output list. This continues until the heap is empty.

### Your output

- Put the output you got for the prefixes provided here

the, thou, thee, that, thag, though, there, their, thought, through



## Experimental
- Explain here what differences did you see in the suggestions generated when you used BFS vs DFS vs UCS.

Since I added nodes in alphabetical order, BFS returned the words in shortlex order (ordered primarily by length and then by alphabetical order) since the shortest words were found by BFS first, while DFS returned the words in simple alphabetical order, disregarding length, since it simply searches all the way down with letters that start at the beginning of the alphabet.







