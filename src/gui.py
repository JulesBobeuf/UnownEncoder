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
        btn_load_image = tk.Button(
            self.root,
            text="Load an image",
            command=self.load_image,
        )
        btn_translate_unown_to_roman = tk.Button(
            self.root,
            text="Translate Unown letters to roman letters",
            command=self.translate_unown_to_roman,
        )
        btn_translate_roman_to_unown = tk.Button(
            self.root,
            text="Translate roman letters to unown letters",
            command=self.translate_roman_to_unown,
        )
        lbl_translated_result = tk.Label(self.root, textvariable=self.translated_text)
        btn_load_image.pack(pady=10)
        btn_translate_unown_to_roman.pack(pady=10)
        btn_translate_roman_to_unown.pack(pady=10)
        lbl_translated_result.pack(pady=10)

    def input_text(self):
        ...

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[
            (
                "Image files",
                "*.png;*jpg;*jpeg;*.gif",
            )
        ])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            lbl_image = tk.Label(self.root, image=img)
            lbl_image.image = img
            lbl_image.pack(pady=10)

    def translate_unown_to_roman(self):
        if self.image_path:
            translated_text = translate_unown_to_roman_function(self.image_path)
            self.translated_text.set(translated_text)
        else:
            tk.messagebox.showwarning("Warning", "Please upload an image first.")

    def translate_roman_to_unown(self):
        input_text = self.input_text.get()
        if input_text:
            translated_image_path = translate_roman_to_unown_function(input_text)
            if translated_image_path:
                img = Image.open(translated_image_path)
                img.thumbnail((300, 300))
                img = ImageTk.PhotoImage(img)
                lbl_image = tk.Label(self.root, image=img)
                lbl_image.image = img
                lbl_image.pack(pady=10)
            else:
                tk.messagebox.showwarning("Warning", "The roman to unown translation have been failed.")
        else:
            tk.messagebox.showwarning("Warning", "Please enter roman text to translate.")


def translate_unown_to_roman_function(image_path):
    # TODO Implements yet
    return ""


def translate_roman_to_unown_function(text):
    # TODO Implements yet
    return ""