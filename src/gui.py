import tkinter as tk

import torch
import torchvision
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk

from ViT import ViT
import os


class TranslatorApp:
    """The translator application class."""

    def __init__(self, root):
        """Init the translation application instance."""
        self.root = root
        self.root.title("UnownEncoder")
        # self.root.iconbitmap("")
        self.root.geometry("600x600")
        self.image_path = ""
        self.lbl_image = None
        self.translated_text = tk.StringVar()
        self.input_text = tk.StringVar()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = './model_save.pt'
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
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device(self.device))["model_state_dict"])
        self.model.eval()
        self.vocab = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
            9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q',
            17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y',
            25: 'Z', 26: '?', 27: '!'
        }

    def create_widgets(self):
        """Create the widgets."""
        entry_text = tk.Entry(
            self.root, textvariable=self.input_text, width=30, font=("Helvetica", 12), justify='center')
        entry_text.pack(pady=10)

        btn_translate_roman_to_unown = tk.Button(
            self.root,
            text="Translate text to unown letters",
            command=self.translate_text_to_unown,
        )
        btn_translate_roman_to_unown.pack(pady=10)

        lbl_title = tk.Label(
            self.root, text="Unown Letters Application", font=("Helvetica", 20, "bold"), fg="#eb4034")
        lbl_title.pack(pady=50)

        btn_load_image = tk.Button(
            self.root,
            text="Load an image",
            command=self.load_image,
        )
        btn_load_image.pack(pady=10)

        btn_translate_unown_to_roman = tk.Button(
            self.root,
            text="Translate Unown letters to text",
            command=self.translate_unown_to_roman,
        )
        btn_translate_unown_to_roman.pack(pady=10)

        lbl_translated_result = tk.Label(
            self.root, textvariable=self.translated_text)
        lbl_translated_result.pack(pady=10)

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
            tk.messagebox.showwarning(
                "Warning", "Please upload an image first.")

    def translate_text_to_unown(self):
        """Translate the input text to Unown image."""
        input_text = self.input_text.get()
        if input_text:
            unown_images = []

            for char in input_text.upper():
                if char == ' ':
                    unown_images.append(Image.new("L", (28, 28)))
                else:
                    image_filename = self.get_image_filename(char)
                    image_path = os.path.join(
                        "./data/images-from-dataloader/", image_filename)

                    if os.path.exists(image_path):
                        unown_images.append(Image.open(image_path))

            if unown_images:
                final_width = len(unown_images) * 28
                final_height = 28

                final_image = Image.new("L", (final_width, final_height))

                x_position = 0
                for unown_img in unown_images:
                    final_image.paste(unown_img, (x_position, 0))
                    x_position += 28

                final_image.show()

                # Save image
                save_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                    title=f"Save Unown Image - {input_text}",
                    initialfile=f"{input_text}_unown_image.png"
                )

                if save_path:
                    final_image.save(save_path)

            else:
                tk.messagebox.showwarning(
                    "Warning", "No images found for the input text.")
        else:
            tk.messagebox.showwarning(
                "Warning", "Please enter text to translate.")

    def get_image_filename(self, char):
        """Get the filename for the Unown image corresponding to the character."""
        if char == "?":
            return "Question.png"
        elif char == "!":
            return "Exclamation.png"
        elif char.isalpha() and char.isupper():
            return f"{char}.png"
        else:
            return None

    def predict_letter(self):
        """Predict letter."""
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.Normalize(0.5, std=0.5)
        ])
        image = Image.open(self.image_path).convert('L')
        width, height = image.size
        numpy_image = np.array(image)
        
        # Add channel dimension for grayscale image
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        
        tensor_image = torch.from_numpy(numpy_image).float()
        result = ""

        # every image is of width 28. We can logically split it every 28 pixels
        for i in range(0, width // 28, 1):
            crt_img = tensor_image[:, 28 * i:(i + 1) * 28]
            crt_img = crt_img.permute(2, 0, 1)  # Change channel order to (C, H, W)
            crt_img = crt_img.unsqueeze(0)
            crt_img = transform(crt_img)
            crt_img = crt_img.repeat(4, 1, 1, 1)
            with torch.no_grad():
                prediction = self.model(crt_img.to(self.device))
            predicted_index = round(torch.mean(torch.argmax(prediction, dim=1).float(), dim=0).item())
            result += self.vocab[predicted_index]
        return result
