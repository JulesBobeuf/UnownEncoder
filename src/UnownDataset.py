import torch
import torchvision
import numpy as np

class UnownDataset(torchvision.datasets.ImageFolder):
    def __init__(self, image_path, label_path, transform=None):
        self.data = torch.from_numpy(np.load(image_path))
        self.labels = torch.from_numpy(np.load(label_path)).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx].float()
        sample = sample.unsqueeze(0)
        sample = self.transform(sample)
        return sample, translate_dict[chr(63+self.labels[idx].item())], self.labels[idx]
    
# for some reason indexes are delayed...
translate_dict = {'A': 'O', 
             'B': 'V', 
             'C': 'B', 
             'D': 'I', 
             'E': 'P',
             'F': 'W', 
             'G': 'C', 
             'H': 'J', 
             'I': 'Q', 
             'J': 'X',
             'K': 'D', 
             'L': 'K', 
             'M': 'R', 
             'N': 'Y', 
             'O': 'E',
             'P': 'L', 
             'Q': 'S', 
             'R': 'Z', 
             'S': 'F', 
             'T': 'M',
             'U': 'T', 
             'V': '!', 
             'W': 'G', 
             'X': 'N', 
             'Y': 'U',
             'Z': '?', 
             '?': 'A', 
             '@': 'H'}
