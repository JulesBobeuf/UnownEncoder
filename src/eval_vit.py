import torch            # https://pytorch.org/docs/stable/index.html
import torchvision      # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
from UnownDataset import UnownDataset
from ViT import ViT

if __name__ == "__main__":
    batch_size = 4
    model_path = './model_save.pt'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)), torchvision.transforms.Normalize(0.5, std=0.5)])
    print(transform) 
    
    test_dataset = UnownDataset("./data/X_test.npy", "./data/Y_test.npy", transform=transform)
    print('Number of train images:', len(test_dataset))
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(test_dataloader)
    
    model = ViT(image_size=28, channel_size=1, patch_size=4, embed_size=512, nb_heads=8, classes=28, nb_layers=3, hidden_size=256, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    print(model)
    
    loss_fct = torch.nn.NLLLoss()
    print(loss_fct)
    
    with torch.no_grad():
        model.eval()

        y_test_pred = []
        y_test_true = []

        for batch_idx, (imgs, labels, tensor_labels) in enumerate(test_dataloader):
  
            imgs = imgs.to(device)
            tensor_labels = tensor_labels.to(device)

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
