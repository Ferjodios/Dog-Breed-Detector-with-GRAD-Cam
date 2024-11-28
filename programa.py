import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from IPython.display import display
import pathlib
import shutil
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import cv2
import time


# Definir constantes
N_EPOCHS = 10
BATCH_SIZE = 32
IMG_HEIGHT = 299
IMG_WIDTH = 299

def load_data():
    train_src = 'train'
    test_src = 'test'

    train_labels = pd.read_csv('labels.csv', index_col='id')
    submission = pd.read_csv('sample_submission.csv')

    train_size = len(os.listdir(train_src))
    test_size = len(os.listdir(test_src))

    print(f"Total training images: {train_size}")
    print(f"Total test images: {test_size}")

    target, dog_breeds = pd.factorize(train_labels['breed'], sort=True)
    train_labels['target'] = target

    #print(f"Dog breeds: {dog_breeds}")

    #display(train_labels.head())
    #display(submission.head())

    print("Number of images per class before augmentation:")
    print(train_labels['breed'].value_counts())

    return train_labels, dog_breeds

def prepare_directories(train_labels, dog_breeds):
    data_dir = pathlib.Path('data')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'

    # Se debe hacer solo una vez
    """data_dir.mkdir()
    train_dir.mkdir()
    test_dir.mkdir()
    ## Create a directory for each dog breed classes
    for breed in dog_breeds:
        breed_dir = train_dir / breed
        breed_dir.mkdir()
    for root, dirs, files in os.walk(train_src):
        for file in files:
            imgName = file.split('.')[0]
            src_file = os.path.join(train_src, file)
            destination = os.path.join(train_dir, train_labels.loc[imgName, 'breed'])
            shutil.copy2(src_file, destination)
    shutil.copytree(test_src, test_dir / 'images')"""

    return train_dir, test_dir

"""Exactamente, has entendido correctamente. 
En PyTorch, la augmentación de datos se aplica en tiempo real durante el entrenamiento. 
Esto significa que el número total de imágenes en el conjunto de datos no cambia, 
pero cada vez que se carga una imagen, se aplica una transformación aleatoria diferente. Como resultado, 
las imágenes que se presentan al modelo durante el entrenamiento son diferentes en cada época debido a las transformaciones aleatorias.

Esto ayuda a mejorar la generalización del modelo al simular variaciones en los datos de entrenamiento, 
lo que hace que el modelo sea más robusto y menos propenso a sobreajustarse a los datos de entrenamiento específicos."""
def create_datasets(train_dir, test_dir):
    # Definir transformaciones para augmentación de datos
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    # Definir transformaciones sin augmentación para validación y prueba
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
    ])

    # Crear el dataset de entrenamiento y validación
    full_dataset = datasets.ImageFolder(root=train_dir, transform=data_augmentation)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Obtener los nombres de las clases
    class_names = full_dataset.classes
    #print(f"Number of classes: {len(class_names)}")
    #print(f"Classes: {class_names}")

    # Crear el dataset de prueba
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_names

def visualize_data(train_loader, test_loader, class_names):
    # Mostrar imágenes del dataset de entrenamiento
    plt.figure(figsize=(20, 20))
    for images, labels in train_loader:
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].permute(1, 2, 0).numpy())
            plt.title(class_names[labels[i]])
            plt.axis("off")
        break

    # Mostrar imágenes del dataset de prueba
    plt.figure(figsize=(20, 20))
    for images, _ in test_loader:
        for i in range(16):
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].permute(1, 2, 0).numpy())
            plt.axis("off")
        break

    plt.show()

def visualize_augmentation(train_loader, class_names):
    # Definir augmentación de datos
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((IMG_HEIGHT, IMG_WIDTH), scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    # Visualizar augmentación de datos
    plt.figure(figsize=(10, 10))
    for images, labels in train_loader:
        first_image = images[0]
        first_image_pil = transforms.ToPILImage()(first_image) 
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            augmented_image = data_augmentation(first_image_pil)
            plt.imshow(augmented_image.permute(1, 2, 0).numpy())
            plt.title(class_names[labels[0]])
            plt.axis("off")
        break

    plt.show()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)

        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        val_loss_history.append(val_epoch_loss)
        val_acc_history.append(val_epoch_acc)

        print(f'Epoch {epoch}/{num_epochs - 1}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}')

    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

def grad_cam(model, input_image, target_layer):
    model.eval()
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    def forward_hook(module, input, output):
        activations.append(output)

    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)

    output = model(input_image)
    pred_class = output.argmax(dim=1).item()
    model.zero_grad()
    class_loss = output[0, pred_class]
    class_loss.backward()

    gradients = gradients[0].cpu().data.numpy()[0]
    activations = activations[0].cpu().data.numpy()[0]

    weights = np.mean(gradients, axis=(1, 2))
    cam = np.zeros(activations.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * activations[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam

def visualize_grad_cam(model, test_loader, target_layer, class_names):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    input_image = images[0].unsqueeze(0)

    cam = grad_cam(model, input_image, target_layer)

    input_image = input_image.squeeze().permute(1, 2, 0).cpu().numpy()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title(f"Original Image: {class_names[labels[0]]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(input_image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Grad-CAM: {class_names[labels[0]]}")
    plt.axis("off")
    plt.show()


if __name__ == '__main__':
    train_labels, dog_breeds = load_data()
    train_dir, test_dir = prepare_directories(train_labels, dog_breeds)
    train_loader, val_loader, test_loader, class_names = create_datasets(train_dir, test_dir)
    time_start = time.time()
    
    #visualize_data(train_loader, test_loader, class_names)
    #visualize_augmentation(train_loader, class_names)
    # Cargar el modelo preentrenado ResNet
    base_model = models.resnet50(pretrained=True)

    # Congelar las capas del modelo base
    for param in base_model.parameters():
        param.requires_grad = False

    # Reemplazar la capa final para la clasificación de 120 clases de razas de perros
    num_features = base_model.fc.in_features
    base_model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 120)
    )

    # Compilar el modelo
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(base_model.fc.parameters(), lr=0.001)

    # Entrenar el modelo
    train_loss, val_loss, train_acc, val_acc = train_model(base_model, train_loader, val_loader, criterion, optimizer, num_epochs=N_EPOCHS)

    # Visualizar los resultados
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 1, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.xticks(list(range(N_EPOCHS)))
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0, 1.0])
    plt.title('Training and Validation Loss')
    plt.xticks(list(range(N_EPOCHS)))
    plt.xlabel('Epoch')
    plt.show()

    time_end = time.time()
    target_layer = base_model.layer4[2].conv3  # Última capa convolucional de ResNet50
    visualize_grad_cam(base_model, test_loader, target_layer, class_names)
    time_real_end = time.time()
    print("Tiempo total de ejecución: ", time_end - time_start)
    print("Tiempo total de ejecución con visualización de Grad-CAM: ", time_real_end - time_start)
