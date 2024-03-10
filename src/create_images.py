import torch  # https://pytorch.org/docs/stable/index.html
import torchvision  # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.UnownDataset import UnownDataset
from model.ViT import ViT
from torch.optim.lr_scheduler import StepLR
import os
import string
import utils.constants as constants

def create_images(batch_size, test_dataloader, save_path="data/images-from-dataloader/"):
    os.makedirs(save_path, exist_ok=True)

    alphabet_set = set(string.ascii_uppercase)
    alphabet_set.add("?")
    alphabet_set.add("!")

    for batch_idx, (imgs, labels, tensor_labels) in enumerate(test_dataloader):
        for i in range(batch_size):
            img = imgs[i][0].numpy()
            label = labels[i]

            if label in alphabet_set:
                alphabet_set.remove(label)

                if label == "!" or label == "@":
                    filename = f"Exclamation.png"
                if label == "?":
                    filename = f"Question.png"
                else:
                    filename = f"{label}.png"

                filepath = os.path.join(save_path, filename)

                plt.imsave(filepath, img, cmap='gray')

        if not alphabet_set:
            break


if __name__ == "__main__":

    test_dataset = UnownDataset(
        "./data/X_test.npy", "./data/Y_test.npy", transform=constants.TRANSFORM)
    print('Number of test images:', len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    print(test_dataloader)

    create_images(constants.BATCH_SIZE, test_dataloader,
                  save_path="./data/images-from-dataloader/")
