import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random
import numpy as np
import os
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    # If you are using GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Call the function at the start of your script
set_seed(42)

INPUT_SIZE = 640
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def submit(model_name):
    model = torch.hub.load('pytorch/vision:v0.13.0', model_name, weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 5)
    model = model.to(device)

    # Load the trained model
    model_name = os.path.join('models', model_name)
    model.load_state_dict(torch.load(f'{model_name}_best_model.pth', map_location=torch.device('cpu')))
    model.eval()

    # Inference
    file_indices = []
    classes = []

    test_folder = 'test'
    test_images = [f for f in os.listdir(test_folder) if f.endswith('.jpeg')]

    for image_name in tqdm(test_images, desc="Inference"):
        image_path = os.path.join(test_folder, image_name)

        # Load image and preprocess
        image = to_pil_image(read_image(image_path))
        image = transform(image).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(image)
            pred = output.argmax(dim=1).item()

        # Store results
        file_index = image_name.split('.jpeg')[0]
        file_indices.append(file_index)
        classes.append(pred)

    # Create DataFrame and save to CSV
    submission_df = pd.DataFrame({
        'file_index': file_indices,
        'class': classes
    })

    submission_df.to_csv(f'{model_name}_submission.csv', index=False)
    print("Saved predictions to submission.csv")


for model in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
    submit(model)


# TO DO: Ensemble