import albumentations as A
import cv2
import torch
import os
from PIL import Image
import pandas as pd
import random
import torch.nn as nn
import torch.utils.data.dataloader
import numpy as np 


torch.backends.nnpack.enabled = False
EPOCH = 100  


class facepointsDataset(torch.utils.data.Dataset):
    def __init__(self, root, points, transform=None):
        self.transform = transform
        self.image_files = list(points.keys())
        self.points = points
        self.root=root

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_name = self.image_files[index]
        img_path = os.path.join(self.root, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        keypoints = self.points[img_name]
        keypoints = [(max(0, x), max(0, y)) for x, y in zip(keypoints[::2], keypoints[1::2])]
        
        if self.transform:
            transformed = self.transform(image=image, keypoints=keypoints)
            image = transformed['image']
            keypoints = transformed['keypoints']
        # keypoints = [(x / 224.0, y / 224.0) for x, y in keypoints]
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        keypoints_tensor = torch.tensor(keypoints).view(-1).float()
        
        return image_tensor, keypoints_tensor

class facepointsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), 
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.MaxPool2d(2, 2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14*2*2 * 14, 64),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(64, 28)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.regressor(x)

def compute_mean_std(dataset):
    mean = np.zeros(3)
    std = np.zeros(3)
    count = 0
    for img, _ in dataset:
        img_np = img.numpy()
        img_np = img_np.reshape(3, -1)
        mean += img_np.mean(axis=1)
        std += img_np.std(axis=1)
        count += 1
    mean /= count
    std /= count
    return list(mean), list(std)

def train_detector(dictionary, root, fast_train=False):
    c = 0
    if fast_train:
        EPOCH=1
    if torch.cuda.is_available(): 
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    if fast_train:
        device = "cpu"
    keys = list(dictionary.keys())

    split = int(0.8 * len(keys))
    train_dict = {k: dictionary[k] for k in keys[:split]}
    val_dict = {k: dictionary[k] for k in keys[split:]}
    
    # Compute mean and std from training data
    temp_transform = A.Compose([
        A.Resize(224, 224),
        A.ToFloat(255),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    temp_train_dataset = facepointsDataset(root, train_dict, transform=temp_transform)
    mean, std = compute_mean_std(temp_train_dataset)
    
    # Define transforms with computed mean and std
    transformTr = A.Compose([
        A.Resize(224, 224),
        A.ShiftScaleRotate(rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        #Experimental
        A.RandomBrightnessContrast(p=0.2),
        # A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.3),
        # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
        # A.GaussNoise(std_range=[0.2,0.44],mean_range=(0,0.2),p = 1),
        # A.Affine(
        # scale=(0.7, 1.3),       # Zoom out (70%) or in (130%)
        # rotate=(-30, 30),       # Rotate between -30° and +30°
        # translate_percent=(-0.2, 0.2),  # Shift up to 20% of image width/height
        # shear=(-20, 20),        # Shear the image
        # p=1.0                   # Always apply this transform
        # ),
        # A.Perspective(scale=(0.05, 0.1), p=0.5),
        # A.HorizontalFlip(0.5),
        # A.VerticalFlip(0.5),
        # A.HorizontalFlip(0.5),
        # A.VerticalFlip(0.5),
        ######################
        A.ToFloat(255),
        A.Normalize(mean=mean, std=std, max_pixel_value=1),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    transformVal = A.Compose([
        A.Resize(224, 224),
        A.ToFloat(255),
        A.Normalize(mean=mean, std=std, max_pixel_value=1),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    train_dataset = facepointsDataset(root, train_dict, transformTr)
    val_dataset = facepointsDataset(root, val_dict, transformVal)
    
    if fast_train:
        batch_size = 4
    else:
        batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = facepointsModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=0.0001)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    for epoch in range(EPOCH):
        model.train()
        train_loss = 0.0
        
        for images, targets in train_dataloader:
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_dataloader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, targets).item() * images.size(0)
        
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        if val_loss<best_val_loss:
            c = 0
            best_val_loss = val_loss
            if not(fast_train):
                torch.save(model.state_dict(), "facepoints_model.pt")
        else:
            c+=1
        print(f'Epoch {epoch+1}/{EPOCH} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f}')
        if c == 6: break
    # torch.save(model.state_dict(), "facepoints_model.pt")
    return model

def detect(model_path, images_folder):
    model = facepointsModel().to('cpu')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.ToFloat(255),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=1),
    ])
    
    results = {}
    for img_name in os.listdir(images_folder):
        img_path = os.path.join(images_folder, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        h_orig, w_orig = image.shape[:2]
        
        # Preprocess
        transformed = transform(image=image)
        img_tensor = torch.from_numpy(transformed['image']).permute(2, 0, 1).unsqueeze(0).float()
        
        # Predict
        with torch.no_grad():
            output = model(img_tensor).squeeze().numpy()
        
        # Rescale keypoints to original image size
        keypoints = output.reshape(-1, 2)
        keypoints[:, 0] *= (w_orig / 224)
        keypoints[:, 1] *= (h_orig / 224)
        results[img_name] = keypoints.flatten().tolist()
    
    return results
