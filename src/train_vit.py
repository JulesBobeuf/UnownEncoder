import torch # https://pytorch.org/docs/stable/index.html
import torchvision # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, std=0.5)])
    print(transform) 
    
    train_dataset=torchvision.datasets.ImageFolder(root="./data/", transform=transform)
    test_dataset=torchvision.datasets.ImageFolder(root="./data/", transform=transform)
    print('Number of train images:', len(test_dataset))
    print('Number of test images:', len(train_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True)
    print(train_dataloader)
    print(test_dataloader)
    
    examples = enumerate(train_dataloader)
    batch_idx, data = next(examples)
    print(batch_idx, len(data), type(data))
    
    print('Images:', data[0].shape)
    b, c, h, w = data[0].shape
    print('Batch(s):', b) # 1 images traitées à la fois
    print('Channel(s):', c) # 3 seul canal de couleur car c'est une image en noir et blanc (3 canaux si c'était une image RGB)
    print('Height:', h) # 60 pixels de hauteur
    print('Width:', w) # 53 pixels de largeur
    print('Labels:', data[1].shape) # 1 label
    print(data[1])
    
    fig = plt.figure()
    for i in range(1):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(data[0][i][0], cmap='gray', interpolation='none')
        plt.title(f'Label: {data[1][i]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()