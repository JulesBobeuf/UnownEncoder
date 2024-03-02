import torch # https://pytorch.org/docs/stable/index.html
import torchvision # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UnownDataset import UnownDataset
from ViT import ViT
from torch.optim.lr_scheduler import StepLR

if __name__ == "__main__":
    
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, std=0.5)])
    print(transform) 
    
    test_dataset = UnownDataset(csv_file="./data/test/description.csv", root_dir="./data/test/", transform=transform)
    print('Number of train images:', len(test_dataset))
    
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    print(test_dataloader)
    
    #display_all_images(batch_size, train_dataloader)
    
    
    model = ViT(image_size=64, channel_size=1, patch_size=4, embed_size=64, nb_heads=8, classes=10, nb_layers=7, hidden_size=256, dropout=0.05).to(device)
    model.load_state_dict(state_dict=torch.load('./model_save.pt'))
    print(model)
    
    loss_fct = torch.nn.NLLLoss()
    print(loss_fct)

    # Permet de s'assurer qu'aucun paramètre du modèle ne sera mis à jour
    with torch.no_grad():
        # Permet d'avertir le modèle qu'il s'agit de la phase d'évaluation
        # Certains paramètres agissent différemment selon si il s'agit de la phase d'entrainement ou d'évaluation
        model.eval() # Compléter ici (indice : c'est une fonction simple sans paramètre)

        # Liste pour conserver l'ensemble des prédictions faites
        y_test_pred = []
        # Liste pour conserver l'ensemble des valeurs à prédire
        y_test_true = []

        # Boucle permettant de parcourir l'ensemble des données du DataLoader de test (les 10000 images !)
        for batch_idx, data in enumerate(test_dataloader):          
            # Envoi des données sur le processeur choisi (CPU ou GPU)
            imgs = data[0].to(device)
            label_to_index = {label: index for index, label in enumerate(set(data[1]))}
            # Convert string labels to tensor format
            labels = torch.tensor([label_to_index[label] for label in data[1]]).to(device)

            # Passage du batch d'images dans le modèle ViT conçu
            # On obtient les prédictions directement en sortie
            predictions = model(imgs) # Compléter ici (indice : un exemple de passage d'un batch dans le modèle a été donné à la toute fin du TP5)

            # La variable contient pour chaque image du batch 10 valeurs
            # Chaque valeur correspond à une probabilité pour chacun des chiffres entre 0 et 9
            # L'indice de la probabilité la plus forte correspond au chiffre prédit par le réseau !
            # On ajoute les prédictions et les valeurs à prédire dans les listes correspondantes
            y_test_pred.extend(predictions.detach().max(1)[1].tolist()) # Compléter ici (indice : on veut l'indice de la valeur maximale des éléments du tenseur pour chaque batch, une fonction PyTorch existe pour cela !)
            y_test_true.extend(labels.detach().tolist())

        # Vérification et calcul de la précision du modèle en comparant pour chaque image son label avec la valeur prédite
        nb_imgs = len(y_test_pred)
        total_correct = 0
        for i in range(nb_imgs):
            if y_test_pred[i] == y_test_true[i]:
                total_correct += 1
        accuracy = total_correct * 100 / nb_imgs

        # Affichage du résultat de précision sur le jeu de données d'évaluation
        print(f"Evaluation accuracy: {accuracy} % ({total_correct} / {nb_imgs})")
