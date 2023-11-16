import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from glob import glob
import argparse
import os


def load_model(model_path: str, model_name: str, device: torch.device, num_classes: int) -> nn.Module:
    """
    Loads a PyTorch model from a given path and modifies its last layer based on the model name and task.

    Args:
        model_path (str): Path to the saved PyTorch model.
        model_name (str): Name of the PyTorch model to load.
        device (torch.device): Device to load the model on (CPU or GPU).
        task (str): Task to perform with the model (binary or multiclass).

    Raises:
        NotImplementedError: If the model name is not implemented or the task is not supported.

    Returns:
        nn.Module: The loaded PyTorch model with the last layer modified based on the model name and task.
    """
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name)
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    if device.type == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=device))
    elif device.type == 'cuda':
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
    
    return model


class InferenceDataset(Dataset):
        def __init__(self, images_path, transforms=None, images_format='jpeg'):
            self.images_path = list(glob(os.path.join(images_path, f'*.{images_format}')))
            if transforms:
                self.transforms = transforms

        def __len__(self):
            return len(self.images_path)

        def __getitem__(self, index):
            img_path = self.images_path[index]
            image_name = os.path.basename(img_path)
            with Image.open(img_path) as img:
                if self.transforms:
                    img = self.transforms(img)

            return image_name, img


def prepare_dataloader(image_dir: str, batch_size: int, num_workers: int, images_format: str = 'jpeg') -> DataLoader:
    """
    Prepares a PyTorch DataLoader for inference.

    Args:
        image_dir (str): Path to the directory containing images.
        batch_size (int): Number of images per batch.
        num_workers (int): Number of worker threads to use for loading data.
        images_format (str, optional): Format of the images. Defaults to 'jpeg'.

    Returns:
        DataLoader: PyTorch DataLoader object for inference.
    """
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = InferenceDataset(images_path=image_dir, transforms=transform, images_format=images_format)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader


def predict(models: list, dataloader: DataLoader, device: torch.device, num_classes: int):
    """
    Predicts the class labels for the given input data using the provided models.

    Args:
        models (list): A list of PyTorch models to use for prediction.
        dataloader (DataLoader): A PyTorch DataLoader object containing the input data.
        device (torch.device): The device to use for computation (e.g. 'cpu' or 'cuda').
        num_classes (int): The number of classes in the classification problem.

    Returns:
        Tuple: A tuple containing two lists - the first list contains the file names of the input data,
        and the second list contains the predicted class labels for each input.
    """
    for model in models:
        model.eval()
    
    image_names = []
    
    # Предсказания моделей (класс 0 или 1)
    predictions = []
    # Вероятность принадлежности к предсказанному классу
    probabilities = []
    
    with torch.no_grad():
        for file_names, inputs in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = torch.zeros(inputs.shape[0], num_classes).to(device)
            for model in models:
                outputs += model(inputs)
            outputs /= len(models)
            logits, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(logits.cpu().numpy())
            image_names.extend(file_names)
    return image_names, predictions, probabilities


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument(
        '--task',
        type=str,
        help='Binary or multiclass classification task. Enter either binary or multiclass',
        required=True
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        help='Directory path to the images.',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Number of images to process at once. Default is 8.'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of worker threads to load data. Default is 4.'
    )
    parser.add_argument(
        '--output_name',
        type=str,
        default='labels.csv',
        help='Name of the output file. Default is labels.csv.'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    task = args.task.lower()
    images_dir = args.images_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_name = args.output_name

    # Determine the number of classes based on the task
    if task == 'binary':
        num_classes = 2
    elif task == 'multiclass':
        num_classes = 5
    else:
        raise NotImplementedError(f'Task {task} is not supported')

    # check if if images_dir exists
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f'{images_dir} does not exist')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = prepare_dataloader(images_dir, batch_size, num_workers)

    BINARY_LABELS = {
        0: 'Authentic',
        1: 'Fictitious'
    }

    MULTI_LABELS = {
        0: 'Authentic',
        1: 'Not on the brake stand',
        2: 'From the screen',
        3: 'From the screen + photoshop',
        4: 'Photoshop'
    }


    if task == 'binary':
        model_path_names = [
            ('models/best_model_resnet18.pth', 'resnet18'),
            ('models/efficientnet_b0_best_model.pt', 'efficientnet_b0'),
            ('models/efficientnet_b2_best_model.pt', 'efficientnet_b2'),
            ('models/efficientnet_b3_best_model.pt', 'efficientnet_b3')
        ]

        models = [
            load_model(model_path, model_name, device, num_classes)
            for model_path, model_name in model_path_names
        ]

        image_names, predictions, probabilities = predict(models, dataloader, device, num_classes)
        predictions = [BINARY_LABELS[pred] for pred in predictions]
        
        result_df = pd.DataFrame({
            'image_name': image_names,
            'class': predictions,
            'probability': probabilities
        }) 

    elif task == 'multiclass':
        model_path_names = [
            ('models/efficientnet_b0_multiclass.pth', 'efficientnet_b0'),
            ('models/efficientnet_b1_multiclass.pth', 'efficientnet_b1'),
            ('models/efficientnet_b2_multiclass.pth', 'efficientnet_b2')
        ]

        models = [
            load_model(model_path, model_name, device, num_classes)
            for model_path, model_name in model_path_names
        ]

        binary_predictions = {}
        multiclass_predictions = {}

        for model, (_, model_name) in zip(models, model_path_names):
            file_names, predictions, probabilities = predict([model], dataloader, device, num_classes)

            multiclass_labels = [MULTI_LABELS[pred] for pred in predictions]
            multiclass_predictions[model_name] = multiclass_labels

            binary_labels = [BINARY_LABELS[1] if pred > 0 else BINARY_LABELS[0] for pred in predictions]
            binary_predictions[model_name] = binary_labels
        
        result_df = pd.DataFrame({'image_name': file_names})

        for key, val in binary_predictions.items():
            result_df[key + '_b'] = val

        for key, val in multiclass_predictions.items():
            result_df[key + '_m'] = val
        
    else:
        raise NotImplementedError(f'Task {task} is not supported')
    
    result_df.to_csv(output_name, index=False)

if __name__ == '__main__':
    main()
