import torch # https://pytorch.org/docs/stable/index.html
import torchvision # https://pytorch.org/vision/stable/index.html
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from UnownDataset import UnownDataset
from ViT import ViT
from torch.optim.lr_scheduler import StepLR

def display_all_images(batch_size, train_dataloader):
    rows = 7  # Number of rows in the subplot grid
    cols = 7  # Number of columns in the subplot grid
    fig = plt.figure(figsize=(10, 10))

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
    batch_size = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  
    
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize((64, 64)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, std=0.5)])
    print(transform) 
    
    train_dataset = UnownDataset(csv_file="./data/train/description.csv", root_dir="./data/train/", transform=transform)
    print('Number of test images:', len(train_dataset))
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(train_dataloader)
    
    #display_all_images(batch_size, train_dataloader)
    
    
    model = ViT(image_size=64, channel_size=1, patch_size=4, embed_size=64, nb_heads=8, classes=10, nb_layers=7, hidden_size=256, dropout=0.05).to(device)
    print(model)
    model.load_state_dict(state_dict=torch.load('./model_save.pt'))
    
    loss_fct = torch.nn.NLLLoss()
    print(loss_fct)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
    print(optimizer)
    
    scheduler = StepLR(optimizer, step_size=90, gamma=0.85)  # Adjust step_size and gamma as needed
    print(scheduler)
    
    # Liste pour conserver les valeurs de loss (rappel : on souhaite minimiser la valeur de loss)
    losses = []
    # Liste pour conserver les précisions du modèle (en %)
    accuracies = []
    # Ces listes permettront d'afficher les courbes de loss et de précision après l'entrainement

    # 1 epoch correspond à un passage complet sur l'ensemble des données (les 60000 images !)
    # Le modèle va donc voir chaque image 10 fois
    nb_epochs = 2500
    torch.cuda.empty_cache()
    print("Training starts")
    # Boucle permettant de faire nb_epochs passages sur l'ensemble des données
    for epoch in range(nb_epochs):
        # Passage du modèle en mode entrainement (certains paramètres agissent différemment selon si il s'agit de la phase d'entrainement ou d'évaluation)
        model.train() # Compléter ici (indice : c'est une fonction simple sans paramètre)

        # Variable pour stocker la valeur de loss sur l'ensemble de l'epoch
        epoch_loss = 0

        # Liste pour conserver l'ensemble des prédictions faites durant l'epoch actuelle
        y_pred = []
        # Liste pour conserver l'ensemble des valeurs à prédire durant l'epoch actuelle
        y_true = []

        # Boucle permettant de parcourir l'ensemble des données du DataLoader (les 60000 images !)
        # Chaque itération contient 32 images et labels comme défini lors de la création du DataLoader
        for batch_idx, data in enumerate(train_dataloader):          
  
            # Envoi des données sur le processeur choisi (CPU ou GPU)
            imgs = data[0].to(device)
            label_to_index = {label: index for index, label in enumerate(set(data[1]))}
            # Convert string labels to tensor format
            labels = torch.tensor([label_to_index[label] for label in data[1]]).to(device)
            # Passage du batch d'images dans le modèle ViT conçu
            # On obtient les prédictions directement en sortie (shape : [32, 10])
            predictions = model(imgs) # Compléter ici (indice : un exemple de passage d'un batch dans le modèle a été donné à la toute fin du TP5)
            # Comparaison des prédictions et des labels à l'aide de la fonction objectif
            loss = loss_fct(predictions, labels) # Compléter ici (indice : vous avez simplement besoin de la fonction objectif définie plus haut, et de 2 paramètres)

            # Nettoyage des anciens paramètres de mise à jour calculés
            optimizer.zero_grad()

            # Calcul des paramètres du modèle à mettre à jour (retropropagation du gradient)
            loss.backward() # Compléter ici (indice : c'est une fonction simple sans paramètre)

            # Mise à jour des paramètres du modèle
            optimizer.step() # Compléter ici (indice : c'est une fonction simple sans paramètre)

            # La variable contient pour chaque image du batch 10 valeurs
            # Chaque valeur correspon d à une probabilité pour chacun des chiffres entre 0 et 9
            # L'indice de la probabilité la plus forte correspond au chiffre prédit par le réseau !
            # On ajoute les prédictions et les valeurs à prédire dans les listes correspondantes
            y_pred.extend(predictions.detach().argmax(dim=-1).tolist()) # Compléter ici (indice : on veut l'indice de la valeur maximale des éléments du tenseur pour chaque batch, une fonction PyTorch existe pour cela !)
            y_true.extend(labels.detach().tolist())

            # Ajout de la valeur de loss du batch à la valeur de loss sur l'ensemble de l'epoch
            epoch_loss += loss.item()

        # Ajout de la loss de l'epoch à la liste de l'ensemble des loss
        losses.append(epoch_loss)

        # Vérification et calcul de la précision du modèle en comparant pour chaque image son label avec la valeur prédite
        nb_imgs = len(y_pred)
        total_correct = 0
        for i in range(nb_imgs):
            if y_pred[i] == y_true[i]:
                total_correct += 1
        accuracy = total_correct * 100 / nb_imgs

        # Ajout de la précision à la liste des précisions
        accuracies.append(accuracy)

        # Affichage des résultats pour l'epoch en cours (loss et précision)
        print("----------")
        print("Epoch:", epoch)
        print("Loss:", epoch_loss)
        print(f"Accuracy: {accuracy} % ({total_correct} / {nb_imgs})")
        
        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()
        
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.show()

    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Train accuracy (%)")
    plt.show()
    
    torch.save(model.state_dict(), './model_save.pt')