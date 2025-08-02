"""
Module for loading the TVFace dataset with annotations.
"""
import os
import json
from PIL import Image
from torch.utils.data import Dataset


class TVFaceDataset(Dataset):
    """
    PyTorch Dataset for TVFace images and their annotations.
    """

    def __init__(self, img_dir: str, annotation_path: str, transform=None):
        """
        Args:
            img_dir (str): Directory containing TVFace images (.jpg).
            annotation_path (str): Path to the JSON file of annotations.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load annotation JSON once
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        self.annotations = data.get('labels', {})
        self.ids = list(self.annotations.keys())

    def __len__(self) -> int:
        """
        Returns:
            int: Total number of images in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        """
        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A sample containing image and its metadata.
        """
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")

        # Load image
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        ann = self.annotations[img_id]
        attrs = ann.get('attributes', {})

        # Extract primary attributes by highest probability
        age = max(attrs.get('age', {}), key=attrs.get('age', {}).get)
        gender = max(attrs.get('gender', {}), key=attrs.get('gender', {}).get)
        race = max(attrs.get('race', {}), key=attrs.get('race', {}).get)
        expression = max(attrs.get('expression', {}), key=attrs.get('expression', {}).get)
        pose = attrs.get('pose', {})  # dict: yaw, pitch, roll

        sample = {
            'image': image,
            'label': ann.get('label'),
            'mask': ann.get('mask'),
            'age': age,
            'gender': gender,
            'race': race,
            'expression': expression,
            'pose': pose,
        }
        return sample

