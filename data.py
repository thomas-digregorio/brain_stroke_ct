import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BrainStrokeDataset(Dataset):
    """
    Dataset class for Brain Stroke CT Scans.
    Reads from a CSV containing 'path' and 'binary_label' columns.
    """
    def __init__(self, csv_file, split, root_dir='.', transform=None):
        """
        Args:
            csv_file (string): Path to the splits.csv file.
            split (string): 'train', 'val', or 'test'.
            root_dir (string): Directory with all the images (usually project root).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = pd.read_csv(csv_file)
        # Filter by split
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        self.root_dir = root_dir
        self.transform = transform
        
        # Default Transform (if none provided)
        # 1. Convert to Tensor (0-1 float)
        # 2. Normalize (ImageNet stats are standard for pre-trained models)
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((512, 512)), # Force strict size (found some 584x512 images)
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Path handling: 
        # The CSV has relative paths like "Brain_Stroke_CT_Dataset/Normal/1.png"
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['path'])
        
        # Open Image (Ensure RGB for EfficientNet, even if source is grayscale)
        image = Image.open(img_path).convert('RGB')
        
        # Labels
        label = int(self.df.iloc[idx]['binary_label'])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label
