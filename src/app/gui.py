import tkinter as tk

import torch
import torchvision
import numpy as np
from tkinter import filedialog
from PIL import Image, ImageTk

from model.ViT import ViT
import os

import utils.constants as constants



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
        self.model = ViT(image_size=constants.IMAGE_SIZE, 
                channel_size=constants.CHANNEL_SIZE, 
                patch_size=constants.PATCH_SIZE, 
                embed_size=constants.EMBED_SIZE, 
                nb_heads=constants.NB_HEADS, 
                classes=constants.NB_CLASSES, 
                nb_layers=constants.NB_LAYERS, 
                hidden_size=constants.HIDDEN_SIZE, 
                dropout=constants.DROPOUT).to(constants.DEVICE)
        self.model.load_state_dict(torch.load(constants.MODEL_PATH, map_location=torch.device(constants.DEVICE)))
        self.vocab = {
            0: 'A', 1: 'H', 2: 'O', 3: 'V', 4: 'B', 5: 'I', 6: 'P', 7: 'W', 8: 'C',
            9: 'J', 10: 'Q', 11: 'X', 12: 'D', 13: 'K', 14: 'R', 15: 'Y', 16: 'E',
            17: 'L', 18: 'S', 19: 'Z', 20: 'F', 21: 'M', 22: 'T', 23: '!', 24: 'G',
            25: 'N', 26: 'U', 27: '?'
        } # reminder : classes are in the wrong order in the dataset.
        # Training the model for more than 2 epocs always gives this delayed result.

    def create_widgets(self):
        """Create the widgets."""
        entry_text = tk.Entry(
            self.root, textvariable=self.input_text, width=30, font=("Helvetica", 12), justify='center')
        entry_text.pack(pady=10)

        btn_translate_latin_to_unown = tk.Button(
            self.root,
            text="Translate text to unown letters",
            command=self.translate_text_to_unown,
        )
        btn_translate_latin_to_unown.pack(pady=10)

        lbl_title = tk.Label(
            self.root, text="Unown Letters Application", font=("Helvetica", 20, "bold"), fg="#eb4034")
        lbl_title.pack(pady=50)

        btn_load_image = tk.Button(
            self.root,
            text="Load an image",
            command=self.load_image,
        )
        btn_load_image.pack(pady=10)

        btn_translate_unown_to_latin = tk.Button(
            self.root,
            text="Translate Unown letters to text",
            command=self.translate_unown_to_latin,
        )
        btn_translate_unown_to_latin.pack(pady=10)

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

    def translate_unown_to_latin(self):
        """Translate the unowned letters to latin letters."""
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
                    unown_images.append(Image.new("L", (constants.IMAGE_SIZE, constants.IMAGE_SIZE)))
                else:
                    image_filename = self.get_image_filename(char)
                    image_path = os.path.join(
                        "./data/images-from-dataloader/", image_filename)

                    if os.path.exists(image_path):
                        unown_images.append(Image.open(image_path))

            if unown_images:
                final_width = len(unown_images) * constants.IMAGE_SIZE
                final_height = constants.IMAGE_SIZE

                final_image = Image.new("L", (final_width, final_height))

                x_position = 0
                for unown_img in unown_images:
                    final_image.paste(unown_img, (x_position, 0))
                    x_position += constants.IMAGE_SIZE

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
        image = Image.open(self.image_path).convert('L')
        width, height = image.size
        numpy_image = np.array(image)
        
        # Add channel dimension for grayscale image
        numpy_image = np.expand_dims(numpy_image, axis=-1)
        
        tensor_image = torch.from_numpy(numpy_image).float()
        result = ""

        # every image is of width constants.IMAGE_SIZE. We can logically split it every constants.IMAGE_SIZE pixels
        for i in range(0,width//constants.IMAGE_SIZE,1):
            crt_img = tensor_image[:, constants.IMAGE_SIZE*i:(i+1)*constants.IMAGE_SIZE]
            crt_img = crt_img.permute(2, 0, 1)  # Change channel order to (C, H, W)
            crt_img = crt_img.unsqueeze(0)
            crt_img = constants.TRANSFORM(crt_img)
            crt_img = crt_img.repeat(4, 1, 1, 1)
            with torch.no_grad():
                prediction = self.model(crt_img.to(constants.DEVICE))
            predicted_index = round(torch.mean(torch.argmax(prediction, dim=1).float(), dim=0).item())
            result += self.vocab[predicted_index]
        return result
