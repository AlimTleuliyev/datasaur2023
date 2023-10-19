# Solution for DataSaur 2023 Hackathon by TeamVici

## Overview
Welcome to our comprehensive solution repository for the 2023 DataSaur Hackathon, specifically designed for the [Kaggle competition: Classify images of Automobiles whether they are authentic or fictitious](https://www.kaggle.com/competitions/case3-datasaur-photo/overview). Our innovative approach involves advanced model training and ensemble techniques to accurately distinguish between authentic and fictitious automobile images.

## Methodology

### Training
Our strategy involved the training of multiple sophisticated models, subsequently ensembled for optimal performance. The details of our training process are meticulously documented in 'train.py'. The ensemble model amalgamates the strengths of several networks, namely resnet18, efficientnet_b0, efficientnet_b2, and efficientnet_b3.

#### Performance Metrics
Our model achieved remarkable efficiency and accuracy on the provided test set, as evidenced by the following metrics:
- Accuracy:  97.43%
- Precision:  96.67%
- Recall:  96.67%
- F1 Score:  97.29%

## Getting Started

### Installation
Clone the repository and set up the environment by executing the following commands:
```bash
git clone https://github.com/AlimTleuliyev/datasaur2023.git
cd datasaur2023
pip install -r requirements.txt
```

### Model Weights
To utilize our pre-trained models, follow these steps:
1. Create a 'models' directory within the cloned repository.
2. Access and download the pre-trained weights via [this link](https://drive.google.com/drive/folders/1zzWCFKr0LSLQsirM7jqb9_S5OscJt4f8?usp=sharing).
3. Place the downloaded weights into the 'models' directory.

## Inference

### Data Preparation
For inference, images should be organized in a specific folder structure. Place the images within a subfolder of your main data directory. This enclosing folder's name will be considered as the class name.

**Required Folder Structure:**
```
data_directory_with_images_to_classify
├── arbitrary_class_name
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── ...
```

### Execution
Utilize 'inference.py' for performing inference. The script accepts four arguments:
- `image_dir`: The path to the directory containing images to classify.
- `batch_size`: The batch size utilized during inference.
- `num_workers`: The number of workers for the dataloader.
- `output_name`: The desired name of the output file.

**Command Example:**
```bash
python inference.py --image_dir data_directory_with_images_to_classify --batch_size 32 --num_workers 4 --output_name results.csv
```

Your results will be saved in a structured CSV file with the specified name.

## Contributors

We extend our heartfelt thanks to the incredible team members who contributed their expertise and hard work to make this project a success.

- **Alim Tleuliyev**: [alim.tleuliyev@nu.edu.kz](mailto:alim.tleuliyev@nu.edu.kz)
- **Alikhan Nurkamal**: [alikhan.nurkamal@nu.edu.kz](mailto:alikhan.nurkamal@nu.edu.kz)
- **Beksultan Tleutayev**: [beksultan.tleutayev@nu.edu.kz](mailto:beksultan.tleutayev@nu.edu.kz)

Feel free to reach out to any of the contributors for questions or feedback concerning the project. We are committed to fostering an open, collaborative environment and welcome any contributions or insights from the community.
