import pandas as pd
from PIL import Image
import torchvision

class UnownDataset(torchvision.datasets.ImageFolder):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        img_path = f"{self.root_dir}/{img_name}"
        image = Image.open(img_path).convert('L')  # 'L' mode for grayscale images

        label = self.labels.iloc[idx, 1]  # Assuming label is in the second column of the CSV file

        if self.transform:
            image = self.transform(image)

        return image, label