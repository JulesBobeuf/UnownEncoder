import torch # https://pytorch.org/docs/stable/index.html
import torchvision # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UnownDataset import UnownDataset
from ViT import ViT
from torch.optim.lr_scheduler import StepLR

import constants

def display_all_images(batch_size, train_dataloader):
    rows = 7  # Number of rows in the subplot grid
    cols = 7  # Number of columns in the subplot grid
    fig = plt.figure(figsize=(10, 10))

    incr = 0
    for batch_idx, (imgs, labels, tensor_labels) in enumerate(train_dataloader):
        for i in range(batch_size):
            plt.subplot(rows, cols, batch_idx * batch_size + i + 1)
            plt.tight_layout()
            plt.imshow(imgs[i][0], cmap='gray', interpolation='none')
            plt.title(f'Label: {labels[i]}')
            plt.xticks([])
            plt.yticks([])
        
        incr += 1
        if (incr > 10): 
            break
    plt.show()

if __name__ == "__main__":
    print(constants.TRANSFORM) 
    
    train_dataset = UnownDataset("./data/X_train.npy", "./data/Y_train.npy", transform=constants.TRANSFORM)
    print('Number of train images:', len(train_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    print(train_dataloader)
    
    #display_all_images(batch_size, train_dataloader)
    
    
    model = ViT(image_size=constants.IMAGE_SIZE, 
                channel_size=constants.CHANNEL_SIZE, 
                patch_size=constants.PATCH_SIZE, 
                embed_size=constants.EMBED_SIZE, 
                nb_heads=constants.NB_HEADS, 
                classes=constants.NB_CLASSES, 
                nb_layers=constants.NB_LAYERS, 
                hidden_size=constants.HIDDEN_SIZE, 
                dropout=constants.DROPOUT).to(constants.DEVICE)
    model.load_state_dict(torch.load(constants.MODEL_PATH, map_location=torch.device(constants.DEVICE)))
    print(model)
    
    optimizer = torch.optim.Adam(model.parameters(), 5e-5)
    print(optimizer)
    
    loss_fct = torch.nn.NLLLoss()
    print(loss_fct)
    
    losses = []
    accuracies = []

    torch.cuda.empty_cache()
    print("Training starts")
    # loop allow to go nb_epochs times over the dataset
    for epoch in range(constants.NB_EPOCH):
        model.train()

        epoch_loss = 0

        # predictions
        y_pred = []
        # good predictions
        y_true = []

        for batch_idx, (imgs, labels, tensor_labels) in enumerate(train_dataloader):
            
            imgs = imgs.to(constants.DEVICE)
            tensor_labels = tensor_labels.to(constants.DEVICE)
            
            predictions = model(imgs)
            loss = loss_fct(predictions, tensor_labels)
            
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            y_pred.extend(predictions.detach().argmax(dim=-1).tolist()) 
            y_true.extend(tensor_labels.detach().tolist())

            epoch_loss += loss.item()

        losses.append(epoch_loss)

        nb_imgs = len(y_pred)
        total_correct = 0
        for i in range(nb_imgs):
            if y_pred[i] == y_true[i]:
                total_correct += 1
        accuracy = total_correct * 100 / nb_imgs

        accuracies.append(accuracy)

        print("----------")
        print("Epoch:", epoch)
        print("Loss:", epoch_loss)
        print(f"Accuracy: {accuracy} % ({total_correct} / {nb_imgs})")
        
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.show()

    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.show()
    
    torch.save(model.state_dict(), constants.MODEL_PATH)
