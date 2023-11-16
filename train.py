import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tqdm import tqdm
import random
import numpy as np
import os
import matplotlib.pyplot as plt


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def initialize_model(model_name):
    """Initialize a model from torchhub."""
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    if model_name.startswith('resnet'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name.startswith('efficientnet'):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    else:
        raise NotImplementedError(f'{model_name} not implemented')

    return model


def calculate_metrics(labels, preds):
    """Calculate metrics like accuracy, precision, recall, and F1 score."""
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return acc, precision, recall, f1


def train(model, criterion, optimizer, train_loader, device, epoch):
    """
    Trains the given model for one epoch using the provided criterion, optimizer, and data loader.

    Args:
        model (torch.nn.Module): The model to train.
        criterion (torch.nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        device (torch.device): The device to use for training.
        epoch (int): The current epoch number.

    Returns:
        tuple: A tuple containing the average loss, accuracy, precision, recall, and F1 score for the epoch.
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for data, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())
    acc, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch} | Train Loss: {avg_loss} | F1: {f1} | Accuracy: {acc} | Precision: {precision} | Recall: {recall}")
    return avg_loss, acc, precision, recall, f1


def validate(model, criterion, val_loader, device):
    """
    Validates the performance of the model on the validation set.

    Args:
        model (torch.nn.Module): The model to be validated.
        criterion (torch.nn.Module): The loss function used for validation.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (str): The device to run the validation on.

    Returns:
        tuple: A tuple containing the average validation loss, accuracy, precision, recall, and F1 score.
    """
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    for data, label in tqdm(val_loader, desc="Validating"):
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss = criterion(output, label)
        val_loss += loss.item()
        all_preds.extend(output.argmax(1).cpu().numpy())
        all_labels.extend(label.cpu().numpy())

    acc, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
    avg_loss = val_loss/len(val_loader)
    print(f"Validation Loss: {avg_loss} | F1: {f1} | Accuracy: {acc} | Precision: {precision} | Recall: {recall}")
    return avg_loss, acc, precision, recall, f1


def train_and_evaluate(model, train_loader, val_loader, device, config, model_name):
    """
    Trains and evaluates the given model using the provided train and validation data loaders, on the specified device.
    The training process is controlled by the provided configuration dictionary, and the resulting model is saved
    under the specified model name. The function returns nothing, but saves the training results and the final model
    in the specified directory.

    Args:
    - model: a PyTorch model to be trained and evaluated
    - train_loader: a PyTorch DataLoader object containing the training data
    - val_loader: a PyTorch DataLoader object containing the validation data
    - device: a PyTorch device object specifying the device to be used for training and evaluation
    - config: a dictionary containing the configuration parameters for the training process
    - model_name: a string specifying the name of the model to be saved

    Returns:
    - None
    """


def train_and_evaluate(model, train_loader, val_loader, device, config, model_name):
    if not os.path.exists(model_name):
        os.makedirs(model_name)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['WARMUP_LR'], weight_decay=config['WEIGHT_DECAY'])

    # Learning Rate Scheduler with Warmup
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['EPOCHS'] - config['WARMUP_EPOCHS'], eta_min=0)

    # Training Loop
    best_f1 = 0.0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_precisions = []
    val_precisions = []
    train_recalls = []
    val_recalls = []
    train_f1s = []
    val_f1s = []

    for epoch in range(1, config['EPOCHS']+1):
        if epoch <= config['WARMUP_EPOCHS']:
            # Linear warmup
            new_lr = config['WARMUP_LR'] + (config['LR0'] - config['WARMUP_LR']) * (epoch / config['WARMUP_EPOCHS'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        else:
            # After warmup, we apply cosine annealing. The scheduler step function is called after optimizer.step() in your training function.
            scheduler.step()

        avg_loss, acc, precision, recall, f1 = train(model, criterion, optimizer, train_loader, device, epoch)
        train_losses.append(avg_loss)
        train_accs.append(acc)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1s.append(f1)

        # if using cuda clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            avg_loss, acc, precision, recall, f1 = validate(model, criterion, val_loader, device)
            val_losses.append(avg_loss)
            val_accs.append(acc)
            val_precisions.append(precision)
            val_recalls.append(recall)
            val_f1s.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            print(f'Saving Best Model at Epoch {epoch}')
            torch.save(model.state_dict(), f'{model_name}/{model_name}_best_model.pt')
        else:
            patience_counter += 1
            if patience_counter > config['PATIENCE']:
                print(f'Early Stopping at Epoch {epoch}')
                break

    # save last model
    print('Saving Last Model')
    torch.save(model.state_dict(), f'{model_name}/{model_name}_last_model.pt')

    print('Training Complete for', model_name)
    
    # plot the training and validation graphs
    # We have, loss, accuracy, precision, recall, and F1 score
    plt.figure(figsize=(20, 10))
    epochs = range(1, len(train_losses) + 1)

    plt.subplot(2, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(epochs, train_precisions, label='Train Precision')
    plt.plot(epochs, val_precisions, label='Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epochs')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(epochs, train_f1s, label='Train F1 Score')
    plt.plot(epochs, val_f1s, label='Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{model_name}/{model_name}_training_results.png')
    plt.show()


def main():
    set_seed(42)

    # Конфиги для тренировки модели
    # Укажите здесь путь к тренировочным и валидационным фотографиям
    # TEST_PATH можете оставить пустым...
    config = {
        "TRAIN_PATH": '/kaggle/input/datasaur-train-valid-10/data_10/train',
        "VAL_PATH": '/kaggle/input/datasaur-train-valid-10/data_10/valid',
        "TEST_PATH": '',
        "BATCH_SIZE": 8,
        "INPUT_SIZE": 640,
        "NUM_WORKERS": 2,
        "EPOCHS": 18,
        "WARMUP_LR": 0.00001,
        "LR0": 0.0001,
        "PATIENCE": 6,
        "WARMUP_EPOCHS": 3,
        "WEIGHT_DECAY": 0.0003
    }

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((config['INPUT_SIZE'], config['INPUT_SIZE'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets and DataLoaders
    train_dataset = datasets.ImageFolder(config['TRAIN_PATH'], transform=transform)
    val_dataset = datasets.ImageFolder(config['VAL_PATH'], transform=transform)
    # test_dataset = datasets.ImageFolder(config['TEST_PATH'], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers = config['NUM_WORKERS'])
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers = config['NUM_WORKERS'])
    # test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers = config['NUM_WORKERS'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model_names = ['resnet18', 'resnet34', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3']
    for model_name in model_names:
        model = initialize_model(model_name)
        model = model.to(device)
        train_and_evaluate(model, train_loader, val_loader, device, config, model_name)

        # Весь код ниже нужен для теста. Тест лучше проводить через inference.py
        # load best model
        # model.load_state_dict(torch.load(f'{model_name}/{model_name}_best_model.pt'))
        # model = model.to(device)
        #
        # Testing on Test
        # with torch.no_grad():
        #     all_preds = []
        #     all_labels = []
        #     for data, label in tqdm(test_loader, desc="Testing"):
        #         data, label = data.to(device), label.to(device)
        #         output = model(data)
        #         all_preds.extend(output.argmax(1).cpu().numpy())
        #         all_labels.extend(label.cpu().numpy())
        #     acc, precision, recall, f1 = calculate_metrics(all_labels, all_preds)
        #     print(f"Test Accuracy: {acc} | Precision: {precision} | Recall: {recall} | F1: {f1}")
        #     print('Testing Complete for', model_name)


if __name__ == '__main__':
    main()
