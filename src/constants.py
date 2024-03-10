import torch
import torchvision

#model settings
IMAGE_SIZE:int = 28 #28x28
CHANNEL_SIZE:int = 1
PATCH_SIZE:int = 4
EMBED_SIZE:int = 512
NB_HEADS:int = 8
NB_CLASSES:int = 28
NB_LAYERS:int = 3
HIDDEN_SIZE:int = 256
DROPOUT:float = 0.2

BATCH_SIZE:int = 4
MODEL_PATH:str = './model_save.pt'
NB_EPOCH:int = 3

#transform
TRANSFORM:torchvision.transforms = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)), torchvision.transforms.Normalize(0.5, std=0.5)])

DEVICE:str = torch.device("cuda" if torch.cuda.is_available() else "cpu")
