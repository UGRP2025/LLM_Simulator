import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class ControlBins:
    """
    Manages the discretization of continuous control values (steering, throttle)
    into bins and vice-versa.
    """
    def __init__(self, steering_bins=[-0.5, -0.1, 0.1, 0.5], throttle_bins=[0.2, 0.5]):
        """
        Initializes the bins.
        Args:
            steering_bins (list): Edges for steering bins.
            throttle_bins (list): Edges for throttle bins.
        """
        self.steering_bins = np.array(steering_bins)
        self.throttle_bins = np.array(throttle_bins)
        self.num_steering_bins = len(steering_bins) + 1
        self.num_throttle_bins = len(throttle_bins) + 1
        self.num_bins = self.num_steering_bins * self.num_throttle_bins

    def to_bin(self, steering, throttle):
        """Converts a (steering, throttle) pair to a single class index."""
        steering_bin = np.digitize(steering, bins=self.steering_bins)
        throttle_bin = np.digitize(throttle, bins=self.throttle_bins)
        return steering_bin * self.num_throttle_bins + throttle_bin

    def from_bin(self, bin_index):
        """Converts a class index back to representative (steering, throttle) values."""
        if bin_index < 0 or bin_index >= self.num_bins:
            raise ValueError("Invalid bin index")

        steering_bin = bin_index // self.num_throttle_bins
        throttle_bin = bin_index % self.num_throttle_bins

        # Return the center of the bins
        steer = self._get_bin_center(steering_bin, self.steering_bins)
        throttle = self._get_bin_center(throttle_bin, self.throttle_bins)
        
        return steer, throttle

    def _get_bin_center(self, bin_index, bins):
        if bin_index == 0:
            return bins[0] - 0.1 # Represent the edge
        if bin_index == len(bins):
            return bins[-1] + 0.1 # Represent the edge
        return (bins[bin_index - 1] + bins[bin_index]) / 2.0

class DrivingDataset(Dataset):
    """Dataset for loading driving data (images and control commands)."""
    def __init__(self, data_path, transform=None, control_bins=None):
        self.data_path = data_path
        self.images_path = os.path.join(data_path, 'images')
        self.json_path = os.path.join(data_path, 'json')
        self.transform = transform
        self.control_bins = control_bins if control_bins is not None else ControlBins()

        self.samples = [os.path.join(self.json_path, f) for f in os.listdir(self.json_path) if f.endswith('.json')]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path = self.samples[idx]
        with open(json_path, 'r') as f:
            data = json.load(f)

        steering = data['steering']
        throttle = data['throttle']

        image_path = os.path.join(self.images_path, data['image_filename'])
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.control_bins.to_bin(steering, throttle)

        return image, label


if __name__ == '__main__':
    # Example of how to use the dataset
    data_path = os.path.expanduser('~/f1tenth_data') # Adjust if you used a different path
    if not os.path.exists(data_path):
        print(f"Error: Data path does not exist: {data_path}")
        print("Please run data_collector.py to generate data first.")
    else:
        # Define image transformations
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Create dataset instance
        control_bins = ControlBins()
        dataset = DrivingDataset(data_path=data_path, transform=transform, control_bins=control_bins)

        if len(dataset) > 0:
            print(f"Dataset contains {len(dataset)} samples.")
            
            # Get a sample
            image, label = dataset[0]
            print(f"Sample 0: image shape={image.shape}, label={label}")

            # Convert label back to control values
            steer, throttle = control_bins.from_bin(label)
            print(f"Label {label} corresponds to steer={steer:.2f}, throttle={throttle:.2f}")
            print(f"Total number of control bins: {control_bins.num_bins}")
        else:
            print("Dataset is empty. Please collect data.")
