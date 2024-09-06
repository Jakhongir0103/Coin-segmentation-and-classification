import os
import random
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import cv2
from PIL import Image
from imgaug import augmenters as iaa

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_aug_dirs(base_dir, categories):
    """
    Create directories for augmented images within the specified base directory 
    for each category provided.
    """
    for category in categories:
        os.makedirs(os.path.join(base_dir, category), exist_ok=True)

def augment_image(image):
    """
    Apply data augmentation techniques to an input image using the imgaug library.
    """
    aug = iaa.Sequential([
        # horizontal flip
        iaa.Fliplr(0.5),
        # vertical flip 
        iaa.Flipud(0.5),
        # scaling and rotation
        iaa.Affine(rotate=(-25, 25)),
        # change brightness
        iaa.Multiply((0.8, 1.2)),
        # add Gaussian noise
        iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
        # random crops
        iaa.Crop(percent=(0, 0.1)),
    ])
    return aug.augment_image(image)

def process_images(src_dir, dest_dir):
    """
    Process images from the source directory, apply augmentation, and save both original 
    and augmented images to the destination directory. 
    """
    # Get the category names
    categories = os.listdir(src_dir)

    # Create a directory for each class
    create_aug_dirs(dest_dir, categories)

    for category in categories:
        category_path = os.path.join(src_dir, category)
        dest_category_path = os.path.join(dest_dir, category)

        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            image = cv2.imread(file_path)

            # Save original image
            cv2.imwrite(os.path.join(dest_category_path, filename), image)

            # Generate 5 augmented images
            for i in range(5):
                aug_image = augment_image(image)
                aug_filename = f"{os.path.splitext(filename)[0]}_aug_{i}{os.path.splitext(filename)[1]}"
                cv2.imwrite(os.path.join(dest_category_path, aug_filename), aug_image)

def train_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module, 
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                loss_fn: torch.nn.Module,
                device: torch.device) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train_k_fold(model: torch.nn.Module, 
            train_data_path: str, 
            transform: transforms.Compose, 
            k_folds: int,
            batch_size: int,
            lr: float,
            loss_fn: torch.nn.Module,
            epochs: int,
            device: torch.device) -> Dict[str, List]:
    """Trains and tests a PyTorch model using k fold.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.
    """
        
    kfold = KFold(n_splits=k_folds, shuffle=True)
    train_data = datasets.ImageFolder(train_data_path, transform=transform)

    results_array = []
    best_epoch = 0
    best_validation_accuracey = 0

    for fold, (train_ids, test_ids) in enumerate(kfold.split(train_data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        print('resetting the parameters of the fully connected layer')
        model.fc.reset_parameters()

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_dataloader = torch.utils.data.DataLoader(
                            train_data, 
                            batch_size=batch_size, sampler=train_subsampler)
        test_dataloader = torch.utils.data.DataLoader(
                            train_data,
                            batch_size=batch_size, sampler=test_subsampler)

        # Create the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Create empty results dictionary
        results = {"train_loss": [],
                "train_acc": [],
                "test_loss": [],
                "test_acc": []
        }
        
        # Make sure model on target device
        model = model.to(device)

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):             
            train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device)
            test_loss, test_acc = test_step(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            device=device)

            # Print out what's happening
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f} | test_loss: {test_loss:.4f} | test_acc: {test_acc:.4f}")

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # Track the best epoch
            if(test_acc > best_validation_accuracey):
                best_validation_accuracey = test_acc
                best_epoch = epoch
        
        test_loss, test_accuracy = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn, device=device)
        print('Loss:',round(test_loss,3))
        print('Accuracy:',round(test_accuracy,3))
        save_model(model, "model", f'model_{fold}.pth')
        results_array.append(results)

    # Return the filled results at the end of the folds
    return results_array, best_epoch


def train_final_model(model: torch.nn.Module,
                        train_data_path: str,
                        lr: float,
                        loss_fn: torch.nn.Module,
                        epochs: int,
                        batch_size: int,
                        transform: transforms.Compose,
                        device: torch.device) -> Dict[str, List]:
        """Trains the final deployable model on the whole dataset.
    
        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.
    
        Calculates, prints and stores evaluation metrics throughout.
        """
        # Create empty results dictionary
        results = {
            "train_loss": [],
            "train_acc": []
        }
    
        # Make sure model on target device
        model = model.to(device)

        # Read the data through DataLoader
        train_data = datasets.ImageFolder(train_data_path, transform=transform)
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(model=model,
                                                dataloader=train_dataloader,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                                device=device)

            # Print out what's happening
            print(f"Epoch: {epoch+1} | train_loss: {train_loss:.4f} | train_acc: {train_acc:.4f}")
    
            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)

        return results

def save_model(model: torch.nn.Module,
                target_dir: str,
                model_name: str):
    """
    Saves a PyTorch model to a target directory.
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                        exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
                f=model_save_path)

def load_model(model_class: torch.nn.Module,
                model_path: str,
                out_features: int = 15) -> torch.nn.Module:
    """
    Loads a PyTorch model from a specified path.
    """
    # Ensure the model path is a Path object
    model_path = Path(model_path)
    
    # Check if the model path exists
    if not model_path.is_file():
        raise FileNotFoundError(f"No model found at: {model_path}")
    
    # Instantiate the model
    model = model_class().to(device)
    model.fc = torch.nn.Linear(in_features=2048, out_features=out_features, device=device)
    
    # Load the model state_dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print(f"[INFO] Loaded model from: {model_path}")
    
    return model

def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    transform: torchvision.transforms
):
    """
    Predicts on a target image with a target model.
    """

    # Open image
    img = Image.open(image_path)

    # Make sure the model is on the target device
    model = model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = transform(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(transformed_image.to(device))

    # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # Plot image with predicted label and probability
    plt.figure()
    plt.imshow(img)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_labels = os.listdir(img_dir)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_labels[idx])
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_name)

def pred_and_return_labels(
    model: torch.nn.Module,
    class_names: List[str],
    image_dir: str,
    transform: torchvision.transforms = None
) -> Dict[str, str]:
    """
    Predicts the class of each image in a directory using the provided model.
    """
    # Create dataset and dataloader
    dataset = CustomImageDataset(image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Make sure the model is on the target device
    model = model.to(device)

    # Set model to evaluation mode
    model.eval()

    predictions = {}
    with torch.inference_mode():
        for images, image_names in dataloader:
            # Move images to the target device
            images = images.to(device)

            # Make predictions
            preds = model(images)

            # Convert logits to probabilities
            pred_probs = torch.softmax(preds, dim=1)

            # Get the predicted class label
            pred_labels = torch.argmax(pred_probs, dim=1)

            # Map predictions to class names and store them in the dictionary
            for image_name, pred_label in zip(image_names, pred_labels):
                predicted_class = class_names[pred_label]
                predictions[image_name] = predicted_class

    return predictions


def pred_and_get_label(model, image, device):
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        image = image.unsqueeze(dim=0).to(device)
        output = model(image)
        pred_probs = torch.softmax(output, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
    return pred_label.item()

def evaluate_model(model, folder, device):
    images, true_labels, class_names = load_images_from_folder(folder)
    true_indices = [class_names.index(label) for label in true_labels]
    pred_indices = [pred_and_get_label(model, img, device) for img in images]

    cm = confusion_matrix(true_indices, pred_indices, labels=range(len(class_names)))

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def load_images_from_folder(folder):
    images = []
    labels = []
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])
    class_names = sorted(os.listdir(folder))
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                img = Image.open(img_path).convert("RGB")
                transformed_img = transform(img)
                images.append(transformed_img)
                labels.append(class_name)
    return images, labels, class_names
