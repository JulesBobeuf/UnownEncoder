import tkinter as tk

from gui import TranslatorApp

if __name__ == "__main__":
    root = tk.Tk()
    app = TranslatorApp(root)
    app.create_widgets()
    root.mainloop()
