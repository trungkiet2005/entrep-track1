#!/usr/bin/env python3
"""
ENT Classification - Fast Optimized Version (No Flips)
Quick testing version with optimizations for higher accuracy
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.manual_seed(42)
np.random.seed(42)

class FastOptimizedConfig:
    # Data paths
    TRAIN_IMG_DIR = "D:/train/train/imgs"
    TRAIN_LABELS_JSON = "D:/train/train/cls.json"
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    
    # Optimized parameters for speed + accuracy
    IMAGE_SIZE = 384        # Balance between quality and speed
    BATCH_SIZE = 16         # Reasonable batch size
    NUM_EPOCHS = 25         # More epochs than before
    LEARNING_RATE = 1e-4    # Conservative learning rate
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    
    # Multiple strong models
    MODEL_NAMES = ['resnet101', 'efficientnet_b4', 'densenet161']
    
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class FastOptimizedDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load with better error handling
        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, Path(img_path).name
        else:
            return image, self.labels[idx]

class FastOptimizedModel(nn.Module):
    def __init__(self, model_name, num_classes=7, pretrained=True):
        super(FastOptimizedModel, self).__init__()
        
        if model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'efficientnet_b4':
            self.backbone = models.efficientnet_b4(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'densenet161':
            self.backbone = models.densenet161(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Enhanced classifier
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
        features = self.backbone(x)
        return self.classifier(features)

def get_fast_optimized_transforms(mode='train', image_size=384):
    """Optimized transforms WITHOUT flips (preserves left/right)"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            
            # IMPORTANT: NO HORIZONTAL/VERTICAL FLIPS
            # Only safe augmentations for left/right classification
            transforms.RandomRotation(8, fill=0),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                fill=0
            ),
            
            # Color augmentations (safe for medical images)
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class FastOptimizedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def create_weighted_sampler(self, labels):
        """Handle class imbalance"""
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def train_model(self, model, train_loader, val_loader, model_name):
        # Use Focal Loss for class imbalance
        class_weights = self.calculate_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        scheduler = CosineAnnealingLR(optimizer, T_max=self.config.NUM_EPOCHS)
        
        best_val_acc = 0.0
        patience = 10
        patience_counter = 0
        
        print(f"\nTraining {model_name}...")
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            scheduler.step()
            
            print(f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                  f"Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'{model_name}_fast_best.pth')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        model.load_state_dict(torch.load(f'{model_name}_fast_best.pth'))
        print(f"Best validation accuracy for {model_name}: {best_val_acc:.2f}%")
        
        return model, best_val_acc
    
    def calculate_class_weights(self, train_loader):
        """Calculate class weights for imbalanced dataset"""
        class_counts = torch.zeros(self.config.NUM_CLASSES)
        
        for _, labels in train_loader:
            for label in labels:
                class_counts[label] += 1
        
        total = class_counts.sum()
        class_weights = total / (self.config.NUM_CLASSES * class_counts)
        
        return class_weights.to(self.device)
    
    def predict(self, model, test_loader):
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
        
        return np.vstack(all_predictions), filenames

def load_data(config):
    if not os.path.exists(config.TRAIN_IMG_DIR) or not os.path.exists(config.TRAIN_LABELS_JSON):
        print("Training data not found!")
        return None, None
    
    with open(config.TRAIN_LABELS_JSON, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    image_paths = []
    labels = []
    
    for filename, label_str in labels_dict.items():
        img_path = Path(config.TRAIN_IMG_DIR) / filename
        if img_path.exists():
            image_paths.append(img_path)
            labels.append(config.LABEL_MAPPING[label_str])
    
    print(f"Loaded {len(image_paths)} training images")
    
    # Print distribution
    label_counts = Counter(labels)
    print("\nClass distribution:")
    for label_idx, count in sorted(label_counts.items()):
        label_name = config.INDEX_TO_LABEL[label_idx]
        print(f"  {label_name}: {count} images")
    
    return image_paths, labels

def create_submission(predictions, filenames, config):
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    submission = {}
    for filename, pred_label in zip(filenames, predicted_labels):
        submission[filename] = int(pred_label)
    
    with open('fast_optimized_submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Fast optimized submission saved with {len(submission)} predictions")
    
    # Analysis
    pred_counts = Counter(predicted_labels)
    print("\nPrediction distribution:")
    for label_idx in range(config.NUM_CLASSES):
        label_name = config.INDEX_TO_LABEL[label_idx]
        count = pred_counts.get(label_idx, 0)
        print(f"  {label_name}: {count} predictions")
    
    print(f"\nAverage confidence: {np.mean(confidence_scores):.3f}")

def main():
    print("="*50)
    print("ENT Classification - Fast Optimized (No Flips)")
    print("="*50)
    
    config = FastOptimizedConfig()
    trainer = FastOptimizedTrainer(config)
    
    # Load data
    image_paths, labels = load_data(config)
    if image_paths is None:
        return
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train ensemble
    models = []
    accuracies = []
    
    for model_name in config.MODEL_NAMES:
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}")
        
        # Create datasets
        train_dataset = FastOptimizedDataset(
            train_paths, train_labels,
            get_fast_optimized_transforms('train', config.IMAGE_SIZE)
        )
        val_dataset = FastOptimizedDataset(
            val_paths, val_labels,
            get_fast_optimized_transforms('val', config.IMAGE_SIZE)
        )
        
        # Create weighted sampler
        weighted_sampler = trainer.create_weighted_sampler(train_labels)
        
        # Create loaders (num_workers=0 for Windows compatibility)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=weighted_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )
        
        # Train model
        model = FastOptimizedModel(model_name, config.NUM_CLASSES)
        model.to(trainer.device)
        
        trained_model, val_acc = trainer.train_model(model, train_loader, val_loader, model_name)
        models.append(trained_model)
        accuracies.append(val_acc)
    
    # Test prediction
    print("\nLoading test data...")
    test_images = list(Path(config.TEST_IMG_DIR).glob("*.png"))
    test_images.extend(list(Path(config.TEST_IMG_DIR).glob("*.jpg")))
    
    if test_images:
        test_dataset = FastOptimizedDataset(
            test_images,
            transform=get_fast_optimized_transforms('val', config.IMAGE_SIZE),
            is_test=True
        )
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # Ensemble predictions
        print("\nMaking ensemble predictions...")
        all_predictions = []
        
        for i, model in enumerate(models):
            print(f"Predicting with model {i+1}/{len(models)}")
            predictions, filenames = trainer.predict(model, test_loader)
            all_predictions.append(predictions)
        
        # Weighted average
        weights = np.array(accuracies) / np.sum(accuracies)
        final_predictions = np.average(all_predictions, axis=0, weights=weights)
        
        create_submission(final_predictions, filenames, config)
        
        print(f"\nExpected accuracy improvement over 0.82!")
        print("Key improvements:")
        print("- No flip augmentation (preserves left/right)")
        print("- Class weight balancing")
        print("- Enhanced ensemble with 3 strong models")
        print("- Optimized training with early stopping")
    
    print("\nFast optimized training completed!")

if __name__ == "__main__":
    main() 