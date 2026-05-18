import tkinter as tk
from tkinter import ttk

def create_gui():
    window = tk.Tk()
    window.title("Autocomplete Demo")

    entry = ttk.Entry(window)
    entry.focus()  # Ensure the entry gets focus
    entry.pack()

    # Test keypress event
    def on_keypress(event):
        print(f"Key pressed: {event.char}")

    entry.bind("<KeyPress>", on_keypress)

    window.mainloop()

create_gui()
