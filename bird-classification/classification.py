import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import mobilenet_v2
import pytorch_lightning as pl
from albumentations.pytorch import ToTensorV2
from sklearn.utils import shuffle
import numpy as np
from PIL import Image
import albumentations as A
import cv2
class BirdDataset(Dataset):
    def __init__(self, img_dir, gt_dict, transform=None):
        self.img_dir = img_dir
        self.filenames = list(gt_dict.keys())
        self.labels = list(gt_dict.values())
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.filenames[idx])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        label = self.labels[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label

class BirdDataModule(pl.LightningDataModule):
    def __init__(self, train_gt, train_img_dir, batch_size=32, num_workers=0):
        super().__init__()
        self.train_gt = train_gt
        self.train_img_dir = train_img_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.9),
            A.RandomBrightnessContrast(p=0.5),
            A.ShiftScaleRotate(rotate_limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
            # A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.3),
            # A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            # A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
            A.GaussNoise(std_range=[0.2,0.44],mean_range=(0,0.2),p = 1),
            A.Perspective(scale=(0.05, 0.1), p=0.5),

            #  A.VerticalFlip(0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def setup(self, stage=None):
        full_dataset = BirdDataset(self.train_img_dir, self.train_gt)
        labels = [full_dataset.labels[i] for i in range(len(full_dataset))]
        unique_classes = np.unique(labels)
        train_indices, val_indices = [], []
        
        for cls in unique_classes:
            cls_indices = np.where(np.array(labels) == cls)[0]
            cls_indices = shuffle(cls_indices, random_state=37)
            n_val = int(len(cls_indices) * 0.2)
            train_indices.extend(cls_indices[n_val:])
            val_indices.extend(cls_indices[:n_val])
        train_indices = shuffle(train_indices, random_state=37)
        val_indices = shuffle(val_indices, random_state=37)
        
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        
        self.train_dataset.dataset.transform = self.train_transform
        self.val_dataset.dataset.transform = self.val_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

class BirdClassifier(pl.LightningModule):
    def __init__(self, fast_train = True, num_classes=50, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        if fast_train:
            flag = False
        else:
            flag = True
        self.model = mobilenet_v2(pretrained=flag)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(self.model.last_channel),
            nn.Dropout(0.3),
            nn.Linear(self.model.last_channel, 300),
            nn.BatchNorm1d(300),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(300,num_classes)
        )
        
        for param in self.model.features[:-6].parameters():
            param.requires_grad = False
            
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr,weight_decay=0.00001)

def train_classifier(train_gt, train_img_dir, fast_train=False):
    pl.seed_everything(37)
    # fast_train = False
    
    batch_size = 4 if fast_train else 32
    lr =  1e-4
    epochs = 1 if fast_train else 100
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    
    data_module = BirdDataModule(train_gt, train_img_dir, batch_size=batch_size)
    model = BirdClassifier(fast_train=fast_train,num_classes=50, lr=lr)
    
    trainer = pl.Trainer(
        accelerator=device,
        # devices=1 if device == 'gpu' else None,
        max_epochs=epochs,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False
    )
    
    trainer.fit(model, data_module)
    if not fast_train:
        torch.save(model.state_dict(), "birds_model.pt")
    
    return model

def classify(model_path, test_img_dir):
    model = BirdClassifier(fast_train=True,num_classes=50,lr = 0.00001).to("cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # model = BirdClassifier(fast_train=True,num_classes=50,lr = 0.0001).load_from_checkpoint(
    #     model_path,
    #     map_location=torch.device('cpu')
    # )
    
    model.eval()
    
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    results = {}
    for img_name in os.listdir(test_img_dir):
        img_path = os.path.join(test_img_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        with torch.no_grad():
            augmented = transform(image=image)
            tensor = augmented['image'].unsqueeze(0)
            pred = model(tensor).argmax().item()
            
        results[img_name] = pred
    
    return results