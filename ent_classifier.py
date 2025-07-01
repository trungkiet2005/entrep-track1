#!/usr/bin/env python3
"""
ENT Image Classification Challenge Solution
High-accuracy solution for ENT endoscopy image classification
"""

import os
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ENTConfig:
    """Configuration class for ENT classification"""
    
    # Data paths
    TRAIN_IMG_DIR = "D:/train/train/imgs"
    TRAIN_LABELS_JSON = "D:/train/train/cls.json"
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    TEST_CSV = "D:/Track1_Public/cls.csv"
    
    # Model parameters
    IMAGE_SIZE = 384  # Increased from 640x480 for better efficiency
    BATCH_SIZE = 32  # Increased for RTX A5000 (24GB memory)
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    
    # Label mapping
    LABEL_MAPPING = {
        'nose-right': 0,
        'nose-left': 1,
        'ear-right': 2,
        'ear-left': 3,
        'vc-open': 4,
        'vc-closed': 5,
        'throat': 6
    }
    
    # Reverse mapping for predictions
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class ENTDataset(Dataset):
    """Custom dataset for ENT images"""
    
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, Path(img_path).name
        else:
            label = self.labels[idx]
            return image, label

class ENTModel(nn.Module):
    """Enhanced model with attention"""
    
    def __init__(self, model_name='resnet101', num_classes=7, pretrained=True):
        super(ENTModel, self).__init__()
        
        # Load backbone
        if model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Enhanced classifier with dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        # Classification
        output = self.classifier(features)
        return output

class DataManager:
    """Data loading and preprocessing manager"""
    
    def __init__(self, config):
        self.config = config
        
    def get_transforms(self, mode='train'):
        """Get augmentation transforms"""
        
        if mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.RandomRotation(15),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def load_training_data(self):
        """Load and prepare training data"""
        
        # Check if data exists
        if not os.path.exists(self.config.TRAIN_IMG_DIR):
            print(f"Warning: Training image directory not found: {self.config.TRAIN_IMG_DIR}")
            return None, None
            
        if not os.path.exists(self.config.TRAIN_LABELS_JSON):
            print(f"Warning: Training labels file not found: {self.config.TRAIN_LABELS_JSON}")
            return None, None
        
        # Load labels
        with open(self.config.TRAIN_LABELS_JSON, 'r') as f:
            labels_dict = json.load(f)
        
        # Prepare data
        image_paths = []
        labels = []
        
        for filename, label_str in labels_dict.items():
            img_path = Path(self.config.TRAIN_IMG_DIR) / filename
            if img_path.exists():
                image_paths.append(img_path)
                labels.append(self.config.LABEL_MAPPING[label_str])
        
        print(f"Loaded {len(image_paths)} training images")
        
        # Print class distribution
        label_counts = Counter(labels)
        print("\nClass distribution:")
        for label_idx, count in sorted(label_counts.items()):
            label_name = self.config.INDEX_TO_LABEL[label_idx]
            print(f"  {label_name} ({label_idx}): {count} images")
        
        return image_paths, labels
    
    def load_test_data(self):
        """Load test data"""
        
        if not os.path.exists(self.config.TEST_IMG_DIR):
            print(f"Warning: Test image directory not found: {self.config.TEST_IMG_DIR}")
            return []
        
        # Get all test images
        test_images = list(Path(self.config.TEST_IMG_DIR).glob("*.png"))
        test_images.extend(list(Path(self.config.TEST_IMG_DIR).glob("*.jpg")))
        test_images.extend(list(Path(self.config.TEST_IMG_DIR).glob("*.jpeg")))
        
        print(f"Found {len(test_images)} test images")
        return test_images

class Trainer:
    """Model training and evaluation"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # GPU optimizations
        if torch.cuda.is_available():
            print(f"GPU Name: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            # Enable automatic mixed precision for faster training
            torch.backends.cudnn.benchmark = True
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
            except:
                pass
        
    def train_model(self, model, train_loader, val_loader):
        """Train model"""
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), 
                               lr=self.config.LEARNING_RATE,
                               weight_decay=self.config.WEIGHT_DECAY)
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.NUM_EPOCHS)
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
        
        # Load best model
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        return model, best_val_acc
    
    def predict(self, model, test_loader):
        """Make predictions"""
        
        model.eval()
        all_predictions = []
        filenames = []
        
        with torch.no_grad():
            for images, names in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                predictions = F.softmax(outputs, dim=1).cpu().numpy()
                all_predictions.append(predictions)
                filenames.extend(names)
        
        all_predictions = np.vstack(all_predictions)
        return all_predictions, filenames

def create_submission(predictions, filenames, config):
    """Create submission file"""
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Create submission dictionary
    submission = {}
    for filename, pred_label in zip(filenames, predicted_labels):
        submission[filename] = int(pred_label)
    
    # Save submission
    with open('submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Submission saved with {len(submission)} predictions")
    
    # Print prediction distribution
    pred_counts = Counter(predicted_labels)
    print("\nPrediction distribution:")
    for label_idx in range(config.NUM_CLASSES):
        label_name = config.INDEX_TO_LABEL[label_idx]
        count = pred_counts.get(label_idx, 0)
        print(f"  {label_name} ({label_idx}): {count} predictions")

def main():
    """Main training and prediction pipeline"""
    
    config = ENTConfig()
    data_manager = DataManager(config)
    trainer = Trainer(config)
    
    # Load training data
    print("Loading training data...")
    image_paths, labels = data_manager.load_training_data()
    
    if image_paths is None:
        print("Could not load training data. Creating dummy predictions for test data.")
        
        # Load test data
        test_images = data_manager.load_test_data()
        if test_images:
            # Create dummy predictions (most frequent class)
            submission = {}
            for img_path in test_images:
                submission[img_path.name] = 0  # nose-right (most frequent)
            
            with open('submission.json', 'w') as f:
                json.dump(submission, f, indent=2)
            print(f"Created dummy submission with {len(submission)} predictions")
        
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = ENTDataset(train_paths, train_labels, 
                             data_manager.get_transforms('train'))
    val_dataset = ENTDataset(val_paths, val_labels, 
                           data_manager.get_transforms('val'))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                          shuffle=False, num_workers=4)
    
    # Train ensemble models
    models = []
    model_names = ['resnet101', 'efficientnet_b4']
    
    for model_name in model_names:
        print(f"\nTraining {model_name}...")
        model = ENTModel(model_name, config.NUM_CLASSES)
        model.to(trainer.device)
        
        trained_model, val_acc = trainer.train_model(model, train_loader, val_loader)
        models.append(trained_model)
        
        # Save model
        torch.save(trained_model.state_dict(), f'{model_name}_best.pth')
    
    # Load test data
    print("\nLoading test data...")
    test_images = data_manager.load_test_data()
    
    if not test_images:
        print("No test images found!")
        return
    
    # Create test dataset
    test_dataset = ENTDataset(test_images, transform=data_manager.get_transforms('val'), 
                            is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                           shuffle=False, num_workers=4)
    
    # Ensemble predictions
    print("\nMaking ensemble predictions...")
    all_predictions = []
    
    for i, model in enumerate(models):
        print(f"Predicting with model {i+1}/{len(models)}")
        predictions, filenames = trainer.predict(model, test_loader)
        all_predictions.append(predictions)
    
    # Average ensemble predictions
    final_predictions = np.mean(all_predictions, axis=0)
    
    # Create submission
    create_submission(final_predictions, filenames, config)
    
    print("\n" + "="*50)
    print("Training and prediction completed!")
    print("Submission file 'submission.json' has been created.")
    print("="*50)

if __name__ == "__main__":
    main() 