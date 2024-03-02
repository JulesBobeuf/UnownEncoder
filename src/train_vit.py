import torch # https://pytorch.org/docs/stable/index.html
import torchvision # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UnownDataset import UnownDataset

def display_all_images(batch_size, train_dataloader):
    rows = 8  # Number of rows in the subplot grid
    cols = 7  # Number of columns in the subplot grid
    fig = plt.figure(figsize=(rows, cols))

    for batch_idx, data in enumerate(train_dataloader):
        #print(f'Batch {batch_idx}, Number of images: {len(data[0])}, Labels: {data[1]}')
        
        """ print('Images:', data[0].shape)
        b, c, h, w = data[0].shape
        print('Batch(s):', b) # 1 images traitées à la fois
        print('Channel(s):', c) # 3 seul canal de couleur car c'est une image en noir et blanc (3 canaux si c'était une image RGB)
        print('Height:', h) # 64 pixels de hauteur
        print('Width:', w) # 64 pixels de largeur
        print('Labels:', data[1].shape) # 1 label
        print(data[1]) """
    
        for i in range(batch_size):
            plt.subplot(rows, cols, batch_idx * batch_size + i + 1)
            plt.tight_layout()
            plt.imshow(data[0][i][0], cmap='gray', interpolation='none')
            plt.title(f'Label: {data[1][i]}')
            plt.xticks([])
            plt.yticks([])
        
    plt.show()

if __name__ == "__main__":
    batch_size = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, std=0.5)])
    print(transform) 
    
    train_dataset = UnownDataset(csv_file="./data/train/description.csv", root_dir="./data/train/", transform=transform)
    test_dataset = UnownDataset(csv_file="./data/test/description.csv", root_dir="./data/test/", transform=transform)
    print('Number of train images:', len(test_dataset))
    print('Number of test images:', len(train_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataloader)
    print(test_dataloader)
    
    display_all_images(batch_size, train_dataloader)