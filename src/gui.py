import tkinter as tk

import torch
import torchvision
from tkinter import filedialog
from PIL import Image, ImageTk

from ViT import ViT


class TranslatorApp:
    """The translator application class."""

    def __init__(self, root):
        """Init the translation application instance."""
        self.root = root
        self.root.title("Image Translator")
        # self.root.iconbitmap("")
        self.root.geometry("600x600")
        self.image_path = ""
        self.lbl_image = None
        self.translated_text = tk.StringVar()
        self.input_text = tk.StringVar()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViT(
            image_size=28,
            channel_size=1,
            patch_size=4,
            embed_size=512,
            nb_heads=8,
            classes=28,
            nb_layers=3,
            hidden_size=256,
            dropout=0.2,
        ).to(self.device)
        self.model.load_state_dict(torch.load('./model_save.pt', map_location=self.device))
        self.model.eval()
        self.vocab = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26: '?', 27: '!'
        }

    def create_widgets(self):
        """Create the widgets."""
        entry_text = tk.Entry(self.root, textvariable=self.input_text, width=30)
        # entry_text.pack(pady=10)
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
        # btn_translate_roman_to_unown.pack(pady=10)
        lbl_translated_result.pack(pady=10)

    def input_text(self):
        ...

    def load_image(self):
        """Load the image."""
        file_path = filedialog.askopenfilename(filetypes=[
            (
                "Image files",
                "*.png ; *jpg ; *jpeg ; *.gif",
            )
        ])
        if file_path:
            if self.lbl_image:
                self.lbl_image.destroy()
            self.image_path = file_path
            img = Image.open(file_path)
            img.thumbnail((300, 300))
            img = ImageTk.PhotoImage(img)
            self.lbl_image = tk.Label(self.root, image=img)
            self.lbl_image.image = img
            self.lbl_image.pack(pady=10)

    def translate_unown_to_roman(self):
        """Translate the unowned letters to roman letters."""
        if self.image_path:
            translated_text = self.predict_letter()
            self.translated_text.set(translated_text)
        else:
            tk.messagebox.showwarning("Warning", "Please upload an image first.")

    def translate_roman_to_unown(self):
        """Translate the roman letters to unowned letters."""
        input_text = self.input_text.get()
        if input_text:
            translated_image_path = input_text
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

    def predict_letter(self):
        """Predict letter."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.Normalize(0.5, std=0.5)
        ])
        image = Image.open(self.image_path).convert('L')
        width, height = image.size
        tensor_image = transform(image)
        result = ""

        # every image is of width 28. We can logically split it every 28 pixels
        for i in range(0,width//28,1):
            crt_img = tensor_image[:, 28*i:(i+1)*28]
            crt_img = crt_img.unsqueeze(0)
            with torch.no_grad():
                prediction = self.model(crt_img.to(self.device))
            predicted_index = torch.argmax(prediction, dim=1).item()
            result += self.vocab[predicted_index]
        return result
