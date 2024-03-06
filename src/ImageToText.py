import torch
from PIL import Image

# Charger l'image
image = Image.open("image.jpg")

# Convertir l'image en tenseur PyTorch
image_tensor = torch.tensor(image)

# Découper l'image en utilisant les indices des pixels
# Par exemple, pour découper un rectangle de coin supérieur gauche (x1, y1) à coin inférieur droit (x2, y2)
x1, y1, x2, y2 = 100, 100, 200, 200
cropped_image_tensor = image_tensor[:, y1:y2, x1:x2]

# Convertir le tenseur en image PIL
cropped_image = Image.fromarray(cropped_image_tensor.numpy())
cropped_image_resized = cropped_image.resize((28, 28))