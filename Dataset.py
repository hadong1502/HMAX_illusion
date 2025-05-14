import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re
import numpy as np
IMG_SIZE = 256

class CrossFinDataset(Dataset):
    """
    Expects filenames like:
      xf_{i}_{label}_{length_case}_{fin_case}_
         {top_length}_{bottom_length}_
         {top_fin_length}_{top_fin_angle_deg}_
         {bottom_fin_length}_{bottom_fin_angle_deg}_
         {top_y}_{bottom_y}.png
    e.g.
      xf_0_1_LONG_DIFF_CONFIG_120_115_15_45_18_50_200.png
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir   = data_dir
        self.transform  = transform

        # compile the exact pattern for your filenames
        self._pattern = re.compile(
            r"^xf_(\d+)_"
            r"([01])_"                            # label
            r"(EQUAL|LONG|SHORT)_"
            r"(SAME_CONFIG|DIFF_CONFIG)_"
            r"(\d+)_(\d+)_"
            r"(\d+)_(\d+)_"
            r"(\d+)_(\d+)_"
            r"(\d+)_(\d+)\.png$"
        )

        self.image_files = []
        self.params      = []

        for fname in os.listdir(data_dir):
            m = self._pattern.match(fname)
            if not m:
                continue
            # build a params dict
            p = {
                'index':           int(m.group(1)),
                'label':           int(m.group(2)),
                'length_case':     m.group(3),
                'fin_case':        m.group(4),
                'top_length':      int(m.group(5)),
                'bottom_length':   int(m.group(6)),
                'top_fin_length':  int(m.group(7)),
                'top_fin_angle':   int(m.group(8)),
                'bot_fin_length':  int(m.group(9)),
                'bot_fin_angle':   int(m.group(10)),
                'top_y':           int(m.group(11)),
                'bottom_y':        int(m.group(12)),
                'filename':        fname,
            }
            self.image_files.append(fname)
            self.params.append(p)

        if not self.image_files:
            raise RuntimeError(f"No valid CrossFin images found in {data_dir}")

        print(f"Found {len(self.image_files)} CrossFin images in {data_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        fname = self.image_files[idx]
        p     = self.params[idx]

        # load image
        path = os.path.join(self.data_dir, fname)
        img  = Image.open(path).convert('L')   # single-channel

        if self.transform:
            img = self.transform(img)

        # return exactly (image, label_tensor)
        label_tensor = torch.tensor(p['label'], dtype=torch.long)
        return img, label_tensor
    
class MullerLyerDataset(Dataset):
    """PyTorch Dataset for the Cross Fin (XF) images.
    Images are in a specified directory and filenames in the format:
    "ml_{i}_{label}_{top_dir_case}_{bottom_dir_case}_{fin_case}_{shaft_length}_{top_fin_length}_{top_fin_angle_deg}_{bottom_fin_length}_{bottom_fin_angle_deg}_{top_y}_{bottom_y}.png"
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        pattern = re.compile(
            r"^ml_(\d+)_"                           # image index
            r"([01])_"                              # label
            r"(LONG|SHORT)_"                        # top_dir_case
            r"(DIFF_DIR|SAME_DIR)_"                 # bottom_dir_case
            r"(SAME_CONFIG|DIFF_CONFIG)_"           # fin_case
            r"(\d+)_"                               # shaft_length
            r"(\d+)_(\d+)_"                         # top_fin_length, top_fin_angle_deg
            r"(\d+)_(\d+)_"                         # bottom_fin_length, bottom_fin_angle_deg
            r"(\d+)_(\d+)\.png$"                    # top_y, bottom_y
        )

        self.image_files = []
        self.image_params = []

        print(f"Scanning directory: {data_dir}")
        for filename in os.listdir(data_dir):
            match = pattern.match(filename)
            if match:
                try:
                    # Extract parameters
                    params = {
                            'index':        int(match.group(1)),
                            'label':        int(match.group(2)),
                            'top_dir':      match.group(3),
                            'bottom_dir':   match.group(4),
                            'fin_case':     match.group(5),
                            'shaft_len':    int(match.group(6)),
                            'top_fin_len':  int(match.group(7)),
                            'top_fin_ang':  int(match.group(8)),
                            'bot_fin_len':  int(match.group(9)),
                            'bot_fin_ang':  int(match.group(10)),
                            'top_y':        int(match.group(11)),
                            'bottom_y':     int(match.group(12)),
                            'filename':     filename
                    }
                    
                    self.image_files.append(filename)
                    self.image_params.append(params)

                except Exception as e:
                    print(f"Warning: Error parsing parameters from {filename}: {e}")
            # else:
            #    print(f"Warning: Skipping file with non-matching name: {filename}")

        if not self.image_files:
            raise RuntimeError(f"No valid Müller-Lyer image files found in {data_dir} matching the pattern.")

        print(f"Found {len(self.image_files)} Müller-Lyer images.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        params = self.image_params[idx]

        try:
            image = Image.open(img_path).convert('L') # Load grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Returning dummy.")
            image = Image.new("L", (IMG_SIZE, IMG_SIZE), 128) # Gray dummy

        if self.transform:
            image = self.transform(image)

        # Return image tensor and the parameter dictionary
        return image, params