import heapq

class TrieNode:
    def __init__(self, char = None):
        self.children = {}  # Dictionary to hold child nodes
        self.is_end_of_word = False  # Flag to indicate complete word
        self.char = char

    # Comparison methods if we will somehow require comparison
    # Generally, we won't need these for the current application
    def __lt__(self, other):
        # Lower comparison can be by the order of the character (if required)
        return (self.char < other.char)
    
    def __gt__(self, other):
        # Lower comparison can be by the order of the character (if required)
        return (self.char > other.char)

    def __eq__(self, other):
        return (self.char == other.char)  # Simple equality check

class Trie:
    def __init__(self):
        self.root = TrieNode()  # Root of the Trie

    def insert(self, word):
        """ Inserts a word into the Trie. """
        current_node = self.root
        for char in word:
            if char not in current_node.children:
                current_node.children[char] = TrieNode(char)  # Create a new child node if it doesn't exist
            current_node = current_node.children[char]
        current_node.is_end_of_word = True  # Mark end of the word

    def find_words_with_prefix_ucs(self, prefix):
        # Returns a list of words in Trie that start with the given prefix using Uniform Cost Search. """
        current_node = self.root
        
        # Traverse the Trie to find the node corresponding to the last character of the prefix.
        for char in prefix:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return []  # If prefix is not found, return an empty list

        # Use UCS to find all words starting with the prefix
        words = []
        priority_queue = [(0, current_node, prefix)]  # (cost, node, current_prefix)

        while priority_queue:
            cost, node, current_prefix = heapq.heappop(priority_queue)

            if node.is_end_of_word:
                words.append((current_prefix, cost))  # Append (word, cost)

            # Add all child nodes to the priority queue
            for char, child_node in node.children.items():
                heapq.heappush(priority_queue, (cost + 1, child_node, current_prefix + char))

        return words

def main():
    words = ["apple", "app", "bat", "batman", "bark"]

    trie = Trie()

    # Insert each word into the Trie
    for word in words:
        trie.insert(word)

    # Example usage: Find words with a given prefix using uniform cost search
    prefix = "ba"
    words_with_prefix_ucs = trie.find_words_with_prefix_ucs(prefix)

    print(f"Words that start with '{prefix}' (UCS):")
    for word, cost in words_with_prefix_ucs:
        print(f"  {word} with cost {cost}")

if __name__ == "__main__":
    main()
