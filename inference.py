import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pandas as pd
import argparse

def load_model(model_path, model_name, device):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights = 'IMAGENET1K_V1')
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    else:
        raise NotImplementedError(f'{model_name} is not implemented')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    # model = model.to(device)
    return model

def prepare_dataloader(image_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(image_dir, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # extract file names from dataset
    image_names = []
    for image_path, _ in dataset.imgs:
        image_name = os.path.basename(image_path)
        image_names.append(image_name)
    
    return image_names, dataloader

# Voting ensemble by probability
def vote_predict(models, dataloader, device):
    """
    This fucntion takes a list of torch models and a dataloader as input
    and returns the predictions of the models by voting.

    Notice: Voting is done by probability. Meaning to say, probabilities of each class across the models are
    summed up, averaged, and the class with the highest probability is chosen.

    Returns a list of predicted classes.
    """
    for model in models:
        model.eval()
    
    predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(dataloader):
            inputs = inputs.to(device)
            outputs = torch.zeros(inputs.shape[0], 2).to(device)
            for model in models:
                outputs += model(inputs)
            outputs /= len(models)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification")
    parser.add_argument(
        '--image_dir',
        type=str,
        help='Directory path to the images.'
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
    
    image_dir = args.image_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    output_name = args.output_name

    model_path_names = [
        ('models/best_model_resnet18.pth', 'resnet18'),
        # ('models/best_model_resnet34.pth', 'resnet34'),
        ('models/efficientnet_b0_best_model.pt', 'efficientnet_b0'),
        # ('models/efficientnet_b1_best_model.pt', 'efficientnet_b1'),
        ('models/efficientnet_b2_best_model.pt', 'efficientnet_b2'),
        ('models/efficientnet_b3_best_model.pt', 'efficientnet_b3')
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    models = [
        load_model(model_path, model_name, device)
        for model_path, model_name in model_path_names
    ]
    
    image_names, dataloader = prepare_dataloader(image_dir, batch_size, num_workers)
    predictions = vote_predict(models, dataloader, device)
    labels = pd.DataFrame({'image_name': image_names, 'class': predictions})
    labels.to_csv(output_name, index=False)

if __name__ == '__main__':
    main()