from collections import defaultdict

def read_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read().lower()
    return text

# Print the frequency tables
def print_table(tables, n):
    n += 1
    for i in range(n):
        print(f"Table {i+1} (n(i_{i+1} | i_{i}, ..., i_1)):")
        for char, prev_chars_dict in tables[i].items():
            for prev_chars, count in prev_chars_dict.items():
                print(f"  P({char} | {prev_chars}) = {count}")
    
    k = 0
    for i in tables:
        print(f"Printing table {k}")
        k += 1
        for j, v in i.items():
            print(j, ' : ', dict(v))