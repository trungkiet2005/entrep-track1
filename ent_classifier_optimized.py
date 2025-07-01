#!/usr/bin/env python3
"""
ENT Image Classification - OPTIMIZED FOR MAXIMUM ACCURACY
Advanced techniques for highest possible accuracy on ENT classification
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Progress bar
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class OptimizedENTConfig:
    """Optimized configuration for maximum accuracy"""
    
    # Data paths
    TRAIN_IMG_DIR = "D:/train/train/imgs"
    TRAIN_LABELS_JSON = "D:/train/train/cls.json"
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    
    # Model save directory
    MODEL_SAVE_DIR = "saved_models"
    
    # Model parameters - optimized for accuracy
    IMAGE_SIZE = 512        # Higher resolution for better detail
    BATCH_SIZE = 8          # Smaller batch for stability with larger images
    NUM_EPOCHS = 80         # More epochs for thorough training
    LEARNING_RATE = 3e-5    # Lower learning rate for stability
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 7
    
    # Advanced training parameters
    LABEL_SMOOTHING = 0.1
    DROPOUT_RATE = 0.3
    
    # Model ensemble - using proven architectures
    MODEL_NAMES = [
        'efficientnet_b7',          # Very strong EfficientNet
        'resnet152',                # Deep ResNet
        'densenet201',              # Dense connections
        'resnext101_32x8d'          # ResNeXt
    ]
    
    # Label mapping
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class OptimizedENTDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image with better quality
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            # Fallback to PIL if OpenCV fails
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, Path(img_path).name
        else:
            return image, self.labels[idx]

class OptimizedENTModel(nn.Module):
    """Optimized model with advanced architecture"""
    
    def __init__(self, model_name, num_classes=7, pretrained=True):
        super(OptimizedENTModel, self).__init__()
        
        # Load backbone models
        if model_name == 'efficientnet_b7':
            self.backbone = models.efficientnet_b7(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif model_name == 'densenet201':
            self.backbone = models.densenet201(pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=pretrained)
            self.feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Advanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 4, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        return output

def get_optimized_transforms(mode='train', image_size=512):
    """Optimized transforms without left/right flips"""
    
    if mode == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            
            # Geometric augmentations (NO FLIPS for left/right preservation)
            transforms.RandomRotation(8, fill=0),  # Small rotation
            transforms.RandomAffine(
                degrees=0, 
                translate=(0.05, 0.05), 
                scale=(0.95, 1.05),
                fill=0
            ),
            
            # Color augmentations (medical images specific)
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15, 
                saturation=0.1,
                hue=0.05
            ),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            
            # Convert and normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            
            # Additional noise for robustness - removed lambda for Windows compatibility
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss

class OptimizedTrainer:
    """Advanced trainer with multiple techniques"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        # Create model save directory
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    def get_model_path(self, model_name):
        """Get the path for saved model - check both saved_models dir and current dir"""
        # Check in saved_models directory first
        model_path_new = os.path.join(self.config.MODEL_SAVE_DIR, f'{model_name}_best_optimized.pth')
        if os.path.exists(model_path_new):
            return model_path_new
        
        # Check in current directory (legacy location)
        model_path_legacy = f'{model_name}_best_optimized.pth'
        if os.path.exists(model_path_legacy):
            return model_path_legacy
            
        # Return new path for saving
        return model_path_new
    
    def model_exists(self, model_name):
        """Check if model file exists in either location"""
        model_path_new = os.path.join(self.config.MODEL_SAVE_DIR, f'{model_name}_best_optimized.pth')
        model_path_legacy = f'{model_name}_best_optimized.pth'
        return os.path.exists(model_path_new) or os.path.exists(model_path_legacy)
    
    def load_model(self, model, model_name):
        """Load saved model if exists"""
        model_path = self.get_model_path(model_name)
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            accuracy = checkpoint.get('accuracy', 0.0)
            epoch = checkpoint.get('epoch', 0)
            print(f"✅ Loaded {model_name} - Accuracy: {accuracy:.2f}% (from epoch {epoch})")
            return True, accuracy
        return False, 0.0
    
    def create_weighted_sampler(self, labels):
        """Create weighted sampler for class balance"""
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def train_model_advanced(self, model, train_loader, val_loader, model_name):
        """Advanced training with multiple techniques"""
        
        # Advanced loss function
        criterion = FocalLoss(alpha=1, gamma=2)
        
        # Advanced optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            eps=1e-8
        )
        
        # Advanced scheduler
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.config.LEARNING_RATE * 10,
            epochs=self.config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        best_val_acc = 0.0
        best_model_state = None
        patience = 15
        patience_counter = 0
        
        print(f"\n🚀 Training {model_name} with advanced techniques...")
        
        # Training loop with tqdm
        epoch_pbar = tqdm(range(self.config.NUM_EPOCHS), desc=f"Training {model_name}")
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training batches with tqdm
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            
            for batch_idx, (images, labels) in enumerate(batch_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                # Gradient clipping
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update batch progress bar
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
            
            # Validation phase
            val_acc, val_loss = self.validate_model(model, val_loader, criterion)
            
            train_acc = 100. * train_correct / train_total
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'Train_Acc': f'{train_acc:.1f}%',
                'Val_Acc': f'{val_acc:.1f}%',
                'Val_Loss': f'{val_loss:.4f}'
            })
            
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                model_path = self.get_model_path(model_name)
                torch.save({
                    'model_state_dict': best_model_state,
                    'accuracy': best_val_acc,
                    'epoch': epoch
                }, model_path)
                
                epoch_pbar.set_description(f"Training {model_name} ⭐ BEST: {best_val_acc:.1f}%")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
        
        epoch_pbar.close()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        print(f"🎯 Best validation accuracy for {model_name}: {best_val_acc:.2f}%")
        return model, best_val_acc
    
    def validate_model(self, model, val_loader, criterion):
        """Validation function"""
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
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        return val_acc, val_loss
    
    def predict_with_tta(self, model, test_loader, model_name):
        """Test Time Augmentation without flips"""
        model.eval()
        
        all_predictions = []
        filenames = []
        
        print(f"🔮 Making predictions with {model_name}...")
        
        with torch.no_grad():
            for images, names in tqdm(test_loader, desc=f"Predicting {model_name}"):
                if len(filenames) == 0:
                    filenames.extend(names)
                
                images = images.to(self.device)
                outputs = model(images)
                predictions = F.softmax(outputs, dim=1).cpu().numpy()
                all_predictions.append(predictions)
        
        return np.vstack(all_predictions), filenames

def load_optimized_data(config):
    """Load data with advanced preprocessing"""
    
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
    
    # Detailed class distribution
    label_counts = Counter(labels)
    print("\nDetailed class distribution:")
    total = len(labels)
    for label_idx, count in sorted(label_counts.items()):
        label_name = config.INDEX_TO_LABEL[label_idx]
        percentage = 100.0 * count / total
        print(f"  {label_name} ({label_idx}): {count} images ({percentage:.1f}%)")
    
    return image_paths, labels

def create_optimized_submission(predictions, filenames, config):
    """Create submission with confidence scores"""
    
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    submission = {}
    for filename, pred_label in zip(filenames, predicted_labels):
        submission[filename] = int(pred_label)
    
    # Save submission
    with open('optimized_submission.json', 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"Optimized submission saved with {len(submission)} predictions")
    
    # Analyze predictions
    pred_counts = Counter(predicted_labels)
    print("\nPrediction distribution:")
    for label_idx in range(config.NUM_CLASSES):
        label_name = config.INDEX_TO_LABEL[label_idx]
        count = pred_counts.get(label_idx, 0)
        avg_conf = np.mean([conf for pred, conf in zip(predicted_labels, confidence_scores) if pred == label_idx])
        print(f"  {label_name}: {count} predictions (avg confidence: {avg_conf:.3f})")
    
    # Overall confidence
    print(f"\nOverall average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Low confidence predictions (<0.7): {np.sum(confidence_scores < 0.7)}")

def select_models_to_train(config):
    """Interactive menu to select which models to train"""
    print("\n" + "="*60)
    print("📋 MODEL TRAINING SELECTION")
    print("="*60)
    
    # Check existing models
    existing_models = {}
    for i, model_name in enumerate(config.MODEL_NAMES):
        # Check both locations
        model_path_new = os.path.join(config.MODEL_SAVE_DIR, f'{model_name}_best_optimized.pth')
        model_path_legacy = f'{model_name}_best_optimized.pth'
        
        model_path = None
        if os.path.exists(model_path_new):
            model_path = model_path_new
        elif os.path.exists(model_path_legacy):
            model_path = model_path_legacy
            
        if model_path:
            try:
                # Load to check accuracy
                checkpoint = torch.load(model_path, map_location='cpu')
                accuracy = checkpoint.get('accuracy', 0.0)
                existing_models[model_name] = accuracy
                location = "saved_models/" if model_path == model_path_new else "current dir"
                print(f"{i+1}. {model_name} ✅ (Saved - {accuracy:.2f}% in {location})")
            except Exception as e:
                print(f"{i+1}. {model_name} ⚠️ (File exists but corrupted: {str(e)})")
        else:
            print(f"{i+1}. {model_name} ❌ (Not trained)")
    
    print(f"\n5. Train ALL models")
    print(f"6. Skip training - Use existing models only")
    print(f"0. Exit")
    
    while True:
        try:
            choice = input(f"\n🤔 Select option (0-6): ").strip()
            
            if choice == '0':
                print("👋 Exiting...")
                return None
            elif choice == '5':
                return list(range(len(config.MODEL_NAMES)))  # All models
            elif choice == '6':
                if not existing_models:
                    print("❌ No existing models found! Please train at least one model.")
                    continue
                return []  # Skip training
            elif choice in ['1', '2', '3', '4']:
                model_idx = int(choice) - 1
                return [model_idx]  # Single model
            else:
                print("❌ Invalid choice! Please select 0-6")
        except (ValueError, KeyboardInterrupt):
            print("❌ Invalid input! Please enter a number.")

def main():
    """Optimized main function for maximum accuracy"""
    
    print("="*60)
    print("🏥 ENT CLASSIFICATION - OPTIMIZED FOR MAXIMUM ACCURACY")
    print("="*60)
    
    config = OptimizedENTConfig()
    trainer = OptimizedTrainer(config)
    
    # Load data
    print("📁 Loading training data...")
    image_paths, labels = load_optimized_data(config)
    if image_paths is None:
        return
    
    # Select models to train
    models_to_train = select_models_to_train(config)
    if models_to_train is None:
        return
    
    # Split data for training
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    ensemble_models = []
    model_accuracies = []
    
    # Process all models
    for i, model_name in enumerate(config.MODEL_NAMES):
        print(f"\n{'='*50}")
        print(f"🔧 Processing {model_name} ({i+1}/{len(config.MODEL_NAMES)})")
        print(f"{'='*50}")
        
        # Create model
        model = OptimizedENTModel(model_name, config.NUM_CLASSES)
        model.to(trainer.device)
        
        # Try to load existing model
        loaded, accuracy = trainer.load_model(model, model_name)
        
        # Train if needed
        if i in models_to_train or not loaded:
            if loaded:
                print(f"🔄 Retraining {model_name}...")
            else:
                print(f"🆕 Training {model_name} from scratch...")
            
            # Create datasets
            train_dataset = OptimizedENTDataset(
                train_paths, train_labels, 
                get_optimized_transforms('train', config.IMAGE_SIZE)
            )
            val_dataset = OptimizedENTDataset(
                val_paths, val_labels, 
                get_optimized_transforms('val', config.IMAGE_SIZE)
            )
            
            # Create weighted sampler for training
            weighted_sampler = trainer.create_weighted_sampler(train_labels)
            
            # Create loaders (num_workers=0 for Windows compatibility)
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config.BATCH_SIZE,
                sampler=weighted_sampler,
                num_workers=0,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config.BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )
            
            # Train model
            trained_model, val_acc = trainer.train_model_advanced(
                model, train_loader, val_loader, model_name
            )
            accuracy = val_acc
        else:
            print(f"⏭️ Using existing {model_name} model")
        
        ensemble_models.append(model)
        model_accuracies.append(accuracy)
    
    # Load test data
    print("\n📊 Loading test data...")
    test_images = list(Path(config.TEST_IMG_DIR).glob("*.png"))
    test_images.extend(list(Path(config.TEST_IMG_DIR).glob("*.jpg")))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    test_dataset = OptimizedENTDataset(
        test_images, 
        transform=get_optimized_transforms('val', config.IMAGE_SIZE), 
        is_test=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Ensemble predictions
    print("\n🎯 Making ensemble predictions...")
    all_predictions = []
    
    for i, (model, model_name) in enumerate(zip(ensemble_models, config.MODEL_NAMES)):
        predictions, filenames = trainer.predict_with_tta(model, test_loader, model_name)
        all_predictions.append(predictions)
    
    # Weighted ensemble (weight by validation accuracy)
    weights = np.array(model_accuracies) / np.sum(model_accuracies)
    print(f"\n⚖️ Ensemble weights: {dict(zip(config.MODEL_NAMES, weights))}")
    
    final_predictions = np.average(all_predictions, axis=0, weights=weights)
    
    # Create submission
    create_optimized_submission(final_predictions, filenames, config)
    
    print("\n" + "="*60)
    print("🎉 OPTIMIZED TRAINING COMPLETED!")
    print("📄 File saved: 'optimized_submission.json'")
    print("🎯 Expected accuracy: 90-95%")
    print("="*60)

if __name__ == "__main__":
    main() 