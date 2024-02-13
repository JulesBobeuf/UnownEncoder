import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk


class TranslatorApp:
    """The translator application class."""

    def __init__(self, root):
        """Init the translation application instance."""
        self.root = root
        self.root.title("Image Translator")
        self.image_path = ""
        self.translated_text = tk.StringVar()
        self.input_text = tk.StringVar()

    def create_widgets(self):
        """Create the widgets."""
        entry_text = tk.Entry(self.root, textvariable=self.input_text, width=30)
        entry_text.pack(pady=10)

    def input_text(self):
        ...
