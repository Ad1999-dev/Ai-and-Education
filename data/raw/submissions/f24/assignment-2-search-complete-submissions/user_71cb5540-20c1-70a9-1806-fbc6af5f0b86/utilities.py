import tkinter as tk
from tkinter import ttk
from autocomplete import Autocomplete

def read_file(filename, autocomplete_engine):
    try:
        with open(filename, 'r') as file:
            document = file.read()
            autocomplete_engine.build_tree(document)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")

def suggest_in_gui(event, entry, listbox, autocomplete_engine):
    prefix = entry.get().lower().split()[-1] if len(entry.get()) > 0 else ''
    print(prefix)
    suggestions = autocomplete_engine.suggest(prefix) 
    listbox.delete(0, tk.END)
    for suggestion in suggestions[:]:
        listbox.insert(tk.END, suggestion)

def create_gui(autocomplete_engine):
    window = tk.Tk()
    window.title("Autocomplete Demo")

    entry = ttk.Entry(window)
    listbox = tk.Listbox(window)

    entry.bind("<KeyRelease>", lambda event: suggest_in_gui(event, entry, listbox, autocomplete_engine))
    entry.pack()
    listbox.pack()

    window.mainloop()