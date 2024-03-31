import torch            # https://pytorch.org/docs/stable/index.html
from torch.utils.data import DataLoader
from model.UnownDataset import UnownDataset
from model.ViT import ViT

import utils.constants as constants

if __name__ == "__main__":
    print(constants.DEVICE)  
    
    test_dataset = UnownDataset("./data/X_test.npy", "./data/Y_test.npy", transform=constants.TRANSFORM)
    print('Number of train images:', len(test_dataset))
    
    test_dataloader = DataLoader(test_dataset, batch_size=constants.BATCH_SIZE, shuffle=True)
    print(test_dataloader)
    
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
    
    with torch.no_grad():
        model.eval()

        y_test_pred = []
        y_test_true = []

        for batch_idx, (imgs, labels, tensor_labels) in enumerate(test_dataloader):
  
            imgs = imgs.to(constants.DEVICE)
            tensor_labels = tensor_labels.to(constants.DEVICE)

            predictions = model(imgs)

            y_test_pred.extend(predictions.detach().max(1)[1].tolist()) 
            y_test_true.extend(tensor_labels.detach().tolist())

        nb_imgs = len(y_test_pred)
        total_correct = 0
        for i in range(nb_imgs):
            if y_test_pred[i] == y_test_true[i]:
                total_correct += 1
        accuracy = total_correct * 100 / nb_imgs

        print(f"Evaluation accuracy: {accuracy} % ({total_correct} / {nb_imgs})")
