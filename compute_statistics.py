"""
Script to compute demographic and pose statistics from TVFaceDataset.
"""
import numpy as np
from collections import Counter, defaultdict
from torch.utils.data import DataLoader

from tvface_dataset import TVFaceDataset


def compute_statistics(dataset):
    """
    Computes distributions for age, gender, race, expression,
    and mean/std for head-pose (yaw, pitch, roll).

    Args:
        dataset (Dataset): Instance of TVFaceDataset.

    Returns:
        dict: Statistics including counters and pose stats.
    """
    age_counter = Counter()
    gender_counter = Counter()
    race_counter = Counter()
    expr_counter = Counter()
    pose_sums = defaultdict(float)
    pose_sqs = defaultdict(float)
    total = len(dataset)

    for sample in dataset:
        age_counter[sample['age']] += 1
        gender_counter[sample['gender']] += 1
        race_counter[sample['race']] += 1
        expr_counter[sample['expression']] += 1
        for axis in ['yaw', 'pitch', 'roll']:
            val = sample['pose'].get(axis, 0.0)
            pose_sums[axis] += val
            pose_sqs[axis] += val ** 2

    # Calculate mean and std for poses
    pose_stats = {}
    for axis in ['yaw', 'pitch', 'roll']:
        mean = pose_sums[axis] / total
        var = (pose_sqs[axis] / total) - (mean ** 2)
        std = np.sqrt(var)
        pose_stats[axis] = {'mean': mean, 'std': std}

    return {
        'age_distribution': age_counter,
        'gender_distribution': gender_counter,
        'race_distribution': race_counter,
        'expression_distribution': expr_counter,
        'pose_statistics': pose_stats,
    }


if __name__ == '__main__':
    import torchvision.transforms as T
    from torch.utils.data import DataLoader

    # Paths (modify as needed)
    IMG_DIR = 'tvface'
    ANNOT_PATH = 'annotation.json'

    # No transforms: only convert to tensor
    transform = T.Compose([T.ToTensor()])

    # Initialize dataset and dataloader
    dataset = TVFaceDataset(img_dir=IMG_DIR, annotation_path=ANNOT_PATH, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    # Compute statistics
    stats = compute_statistics(dataset)

    # Print results
    print("=== Age Distribution ===")
    for k, v in stats['age_distribution'].items():
        print(f"{k}: {v} ({v/len(dataset)*100:.2f}%)")

    print("\n=== Gender Distribution ===")
    for k, v in stats['gender_distribution'].items():
        print(f"{k}: {v} ({v/len(dataset)*100:.2f}%)")

    print("\n=== Race Distribution ===")
    for k, v in stats['race_distribution'].items():
        print(f"{k}: {v} ({v/len(dataset)*100:.2f}%)")

    print("\n=== Expression Distribution ===")
    for k, v in stats['expression_distribution'].items():
        print(f"{k}: {v} ({v/len(dataset)*100:.2f}%)")

    print("\n=== Head-Pose Statistics ===")
    for axis, vals in stats['pose_statistics'].items():
        print(f"{axis}: mean={vals['mean']:.2f}, std={vals['std']:.2f}")