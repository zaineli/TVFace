# TVFace Dataset Usage Guide

This guide explains how to use the TVFace dataset and its Python utilities for loading, processing, and analyzing facial images and annotations.

## Table of Contents
- Installation
- Loading the Dataset
- Sample Structure
- Splitting the Dataset
- Accessing Annotations
- Visualization
- Filtering and Advanced Usage

---

## Installation

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## Loading the Dataset

Use the `TVFaceDataset` class to load images and annotations:

```python
from tvface_dataset import TVFaceDataset
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

dataset = TVFaceDataset(
    img_dir='path/to/tvface_dataset/faces',
    annotation_path='path/to/tvface_dataset/annotations.json',
    transform=transform
)

sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")
print(f"Identity label: {sample['label']}")
print(f"Gender: {sample['gender']}")
print(f"Race: {sample['race']}")
print(f"Age: {sample['age']}")
print(f"Expression: {sample['expression']}")
```

---

## Sample Structure

Each sample is a dictionary with:

| Key         | Description                       |
|-------------|-----------------------------------|
| image       | Image tensor                      |
| label       | Identity label (int)              |
| mask        | Exclusion flag (float)            |
| age         | Age group (str)                   |
| gender      | Gender (str)                      |
| race        | Ethnicity (str)                   |
| expression  | Expression (str)                  |
| pose        | Head pose dict (yaw, pitch, roll) |

---

## Splitting the Dataset

Split into train/val/test sets:

```python
from torch.utils.data import random_split, DataLoader
import torch

generator = torch.Generator().manual_seed(42)
size = len(dataset)
train_size = int(size * 0.7)
val_size = int(size * 0.15)
test_size = size - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

---

## Accessing Annotations

Get full probability distributions for attributes:

```python
img_id = dataset.ids[0]
ann = dataset.annotations[img_id]
age_probs = ann['attributes']['age']
gender_probs = ann['attributes']['gender']
race_probs = ann['attributes']['race']
expression_probs = ann['attributes']['expression']
print("Age probabilities:", age_probs)
print("Gender probabilities:", gender_probs)
```

---

## Visualization

Visualize a sample:

```python
import matplotlib.pyplot as plt

sample = dataset[0]
img = sample['image'].permute(1, 2, 0)
plt.imshow(img)
plt.title(f"ID: {sample['label']}, Gender: {sample['gender']}, Race: {sample['race']}")
plt.axis('off')
plt.show()
```

---

## Filtering and Advanced Usage

### Filter by Demographic

```python
female_indices = [i for i, idx in enumerate(dataset.ids)
                 if max(dataset.annotations[idx]['attributes']['gender'],
                        key=dataset.annotations[idx]['attributes']['gender'].get) == 'Female']
from torch.utils.data import Subset
female_dataset = Subset(dataset, female_indices)
```

### Aggregate Pose Information

```python
import numpy as np
yaw, pitch, roll = [], [], []
for i in range(len(dataset)):
    pose = dataset[i]['pose']
    yaw.append(pose.get('yaw', 0))
    pitch.append(pose.get('pitch', 0))
    roll.append(pose.get('roll', 0))
print(f"Average pose: Yaw={np.mean(yaw):.2f}, Pitch={np.mean(pitch):.2f}, Roll={np.mean(roll):.2f}")
```

### Custom Collate Function

```python
def custom_collate_fn(batch):
    images = torch.stack([item['image'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    metadata = {
        'gender': [item['gender'] for item in batch],
        'race': [item['race'] for item in batch],
        'age': [item['age'] for item in batch],
        'expression': [item['expression'] for item in batch]
    }
    return {'images': images, 'labels': labels, 'metadata': metadata}

loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate_fn)
```

---

For more details, see the full documentation or contact the dataset maintainers.
