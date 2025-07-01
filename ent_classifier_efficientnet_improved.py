#!/usr/bin/env python3
"""
ENT Image Classification - ULTRA HIGH ACCURACY VERSION
Advanced techniques to achieve 96%+ accuracy
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler
import timm

# Progress bar
from tqdm import tqdm

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class UltraConfig:
    """Ultra high-accuracy configuration"""
    
    # Data paths
    TRAIN_IMG_DIR = "D:/train/train/imgs"
    TRAIN_LABELS_JSON = "D:/train/train/cls.json"
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    MODEL_SAVE_DIR = "saved_models"
    
    # Ultra high-performance parameters
    IMAGE_SIZE = 512        # Higher resolution for better detail
    BATCH_SIZE = 8          # Smaller batch for larger models
    NUM_EPOCHS = 100        # More epochs for convergence
    LEARNING_RATE = 1e-5    # Very low LR for stability
    WEIGHT_DECAY = 1e-4     # Strong regularization
    NUM_CLASSES = 7
    
    # Advanced training parameters
    LABEL_SMOOTHING = 0.05  # Reduced for medical data
    DROPOUT_RATE = 0.5
    MIXUP_ALPHA = 0.2
    CUTMIX_PROB = 0.3
    
    # Multi-model ensemble
    USE_ENSEMBLE = True
    ENSEMBLE_MODELS = [
        'efficientnet_b7',
        'convnext_large', 
        'swin_large_patch4_window12_384',
        'beit_large_patch16_384'
    ]
    
    # Test Time Augmentation
    USE_TTA = True
    TTA_CROPS = 5
    
    # Learning rate schedule
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-8
    PATIENCE = 20
    
    # Cross-validation
    USE_KFOLD = True
    N_FOLDS = 5
    
    # Advanced data augmentation
    USE_ADVANCED_AUG = True
    
    # Label mapping
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class UltraAugmentation:
    """Ultra advanced augmentation for medical images"""
    
    @staticmethod
    def get_transforms(mode='train', image_size=512):
        if mode == 'train':
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size + 64, image_size + 64)),
                transforms.RandomCrop((image_size, image_size)),
                
                # Advanced geometric augmentations
                transforms.RandomRotation(8, fill=0),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                    fill=0
                ),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
                
                # Medical-specific augmentations
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.1,
                    hue=0.05
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
                transforms.RandomEqualize(p=0.2),
                
                # Advanced photometric augmentations
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                
                # Advanced random erasing
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
            ])
        else:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    @staticmethod
    def get_tta_transforms(image_size=512):
        """Test Time Augmentation transforms"""
        return [
            # Original
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Horizontal flip
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Slight rotation
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size + 32, image_size + 32)),
                transforms.RandomRotation(5),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Center crop from larger
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
                transforms.CenterCrop((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            # Contrast adjustment
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        ]

class UltraModel(nn.Module):
    """Ultra high-performance model ensemble"""
    
    def __init__(self, num_classes=7, model_name='efficientnet_b7', pretrained=True):
        super(UltraModel, self).__init__()
        
        # Use TIMM for access to latest models
        if model_name == 'efficientnet_b7':
            self.backbone = timm.create_model('efficientnet_b7', pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name == 'convnext_large':
            self.backbone = timm.create_model('convnext_large', pretrained=pretrained)
            self.feature_dim = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Identity()
        elif model_name == 'swin_large_patch4_window12_384':
            self.backbone = timm.create_model('swin_large_patch4_window12_384', pretrained=pretrained)
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif model_name == 'beit_large_patch16_384':
            self.backbone = timm.create_model('beit_large_patch16_384', pretrained=pretrained)
            self.feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            # Fallback to EfficientNet
            self.backbone = timm.create_model('efficientnet_b7', pretrained=pretrained)
            self.feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        
        # Advanced feature processing
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        
        # Multi-head attention with more heads
        self.attention = nn.MultiheadAttention(
            embed_dim=self.feature_dim,
            num_heads=16,
            dropout=0.1,
            batch_first=True
        )
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.feature_dim, self.feature_dim // 16, 1),
            nn.ReLU(),
            nn.Conv1d(self.feature_dim // 16, self.feature_dim, 1),
            nn.Sigmoid()
        )
        
        # Ultra deep classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply layer normalization
        features = self.feature_norm(features)
        
        # Reshape for attention
        features_seq = features.unsqueeze(1)
        
        # Apply self-attention
        attended_features, _ = self.attention(features_seq, features_seq, features_seq)
        attended_features = attended_features.squeeze(1)
        
        # Channel attention
        channel_weights = self.channel_attention(features.unsqueeze(-1)).squeeze(-1)
        features_weighted = features * channel_weights
        
        # Combine all features
        final_features = features + attended_features + features_weighted
        
        # Classification
        output = self.classifier(final_features)
        return output

class UltraLoss(nn.Module):
    """Ultra advanced loss combining multiple objectives"""
    
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.05):
        super(UltraLoss, self).__init__()
        self.focal_loss = self._focal_loss
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def _focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            
        return focal_loss.mean()
    
    def _label_smoothed_ce(self, inputs, targets):
        """Label smoothed cross entropy"""
        log_probs = F.log_softmax(inputs, dim=1)
        targets_smooth = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets_smooth = targets_smooth * (1 - self.label_smoothing) + self.label_smoothing / inputs.size(1)
        loss = -(targets_smooth * log_probs).sum(dim=1).mean()
        return loss
    
    def forward(self, inputs, targets):
        # Combine focal loss and label smoothed CE
        focal = self.focal_loss(inputs, targets)
        smooth_ce = self._label_smoothed_ce(inputs, targets)
        return 0.7 * focal + 0.3 * smooth_ce

class UltraTrainer:
    """Ultra advanced trainer for 96%+ accuracy"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        self.scaler = GradScaler()
        
    def calculate_class_weights(self, labels):
        """Calculate sophisticated class weights"""
        class_counts = Counter(labels)
        total_samples = len(labels)
        
        # Effective number of samples reweighting
        beta = 0.9999
        effective_nums = []
        weights = []
        
        for i in range(self.config.NUM_CLASSES):
            if i in class_counts:
                effective_num = (1.0 - beta**class_counts[i]) / (1.0 - beta)
                weights.append(1.0 / effective_num)
            else:
                weights.append(1.0)
                
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum() * len(weights)
        
        return torch.FloatTensor(weights).to(self.device)
    
    def train_ensemble(self, image_paths, labels):
        """Train ensemble of models"""
        
        if not self.config.USE_ENSEMBLE:
            return self.train_single_model(image_paths, labels, self.config.ENSEMBLE_MODELS[0])
        
        print(f"🎯 Training ensemble of {len(self.config.ENSEMBLE_MODELS)} models...")
        
        ensemble_models = []
        ensemble_accuracies = []
        
        for model_name in self.config.ENSEMBLE_MODELS:
            print(f"\n{'='*60}")
            print(f"🔥 Training {model_name}")
            print(f"{'='*60}")
            
            try:
                model, accuracy = self.train_single_model(image_paths, labels, model_name)
                ensemble_models.append((model, model_name))
                ensemble_accuracies.append(accuracy)
                print(f"✅ {model_name} completed - Accuracy: {accuracy:.2f}%")
            except Exception as e:
                print(f"❌ {model_name} failed: {e}")
                continue
        
        # Calculate ensemble accuracy
        mean_acc = np.mean(ensemble_accuracies)
        print(f"\n🎉 Ensemble Mean Accuracy: {mean_acc:.2f}%")
        
        return ensemble_models, mean_acc
    
    def train_single_model(self, image_paths, labels, model_name):
        """Train single model with k-fold CV"""
        
        if not self.config.USE_KFOLD:
            return self.train_single_fold(image_paths, labels, model_name)
        
        kfold = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True, random_state=42)
        fold_accuracies = []
        best_model = None
        best_accuracy = 0
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths, labels)):
            print(f"\n📁 FOLD {fold + 1}/{self.config.N_FOLDS} - {model_name}")
            
            train_paths = [image_paths[i] for i in train_idx]
            val_paths = [image_paths[i] for i in val_idx]
            train_labels = [labels[i] for i in train_idx]
            val_labels = [labels[i] for i in val_idx]
            
            model, accuracy = self.train_single_fold(
                train_paths, train_labels, val_paths, val_labels, model_name, fold
            )
            
            fold_accuracies.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        
        mean_acc = np.mean(fold_accuracies)
        print(f"📊 {model_name} K-fold Mean: {mean_acc:.2f}% ± {np.std(fold_accuracies):.2f}%")
        
        return best_model, mean_acc
    
    def train_single_fold(self, train_paths, train_labels, val_paths=None, val_labels=None, model_name='efficientnet_b7', fold=None):
        """Train single fold with ultra techniques"""
        
        if val_paths is None:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=0.15, random_state=42, stratify=train_labels
            )
        
        # Create model
        model = UltraModel(
            num_classes=self.config.NUM_CLASSES,
            model_name=model_name
        ).to(self.device)
        
        # Calculate class weights
        class_weights = self.calculate_class_weights(train_labels)
        
        # Ultra loss function
        criterion = UltraLoss(
            alpha=class_weights,
            gamma=2.5,
            label_smoothing=self.config.LABEL_SMOOTHING
        )
        
        # Advanced optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Advanced scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=self.config.WARMUP_EPOCHS
        )
        
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,
            T_mult=2,
            eta_min=self.config.MIN_LR
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.WARMUP_EPOCHS]
        )
        
        # Create datasets
        train_dataset = UltraDataset(
            train_paths, train_labels,
            UltraAugmentation.get_transforms('train', self.config.IMAGE_SIZE)
        )
        
        val_dataset = UltraDataset(
            val_paths, val_labels,
            UltraAugmentation.get_transforms('val', self.config.IMAGE_SIZE)
        )
        
        # Advanced sampler
        train_sampler = self.create_weighted_sampler(train_labels)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            sampler=train_sampler,
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
            prefetch_factor=2
        )
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        suffix = f"_{model_name}_fold{fold}" if fold is not None else f"_{model_name}"
        
        for epoch in range(self.config.NUM_EPOCHS):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
            
            for images, labels in train_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                self.scaler.step(optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%'
                })
            
            # Validation with TTA
            val_acc, val_loss = self.validate_with_tta(model, val_loader, criterion)
            
            scheduler.step()
            
            train_acc = 100. * train_correct / train_total
            
            print(f"Epoch {epoch+1}: Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = os.path.join(
                    self.config.MODEL_SAVE_DIR,
                    f'ultra_model{suffix}_best.pth'
                )
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'model_name': model_name,
                    'accuracy': best_val_acc,
                    'epoch': epoch
                }, model_path)
                patience_counter = 0
                print(f"✨ New best model saved! Accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
        
        return model, best_val_acc
    
    def validate_with_tta(self, model, val_loader, criterion):
        """Validation with Test Time Augmentation"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        if not self.config.USE_TTA:
            # Standard validation
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
        else:
            # TTA validation
            tta_transforms = UltraAugmentation.get_tta_transforms(self.config.IMAGE_SIZE)
            
            with torch.no_grad():
                for original_images, labels in val_loader:
                    labels = labels.to(self.device)
                    
                    # Collect predictions from all TTA transforms
                    all_outputs = []
                    
                    for transform in tta_transforms:
                        # Apply transform to batch
                        tta_images = []
                        for img in original_images:
                            # Convert tensor back to numpy for transform
                            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                            tta_img = transform(img_np)
                            tta_images.append(tta_img)
                        
                        tta_batch = torch.stack(tta_images).to(self.device)
                        outputs = model(tta_batch)
                        all_outputs.append(F.softmax(outputs, dim=1))
                    
                    # Average predictions
                    avg_outputs = torch.stack(all_outputs).mean(dim=0)
                    
                    # Calculate loss with original transform
                    original_images = original_images.to(self.device)
                    loss_outputs = model(original_images)
                    loss = criterion(loss_outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = avg_outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        return val_acc, val_loss
    
    def create_weighted_sampler(self, labels):
        """Create advanced weighted sampler"""
        class_counts = Counter(labels)
        
        # Effective number of samples weighting
        beta = 0.9999
        weights = []
        
        for label in labels:
            effective_num = (1.0 - beta**class_counts[label]) / (1.0 - beta)
            weight = 1.0 / effective_num
            weights.append(weight)
        
        return WeightedRandomSampler(weights, len(weights), replacement=True)

class UltraDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load and preprocess image
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            from PIL import Image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Enhanced preprocessing
        image = cv2.bilateralFilter(image, 9, 75, 75)  # Noise reduction
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, Path(img_path).name
        else:
            return image, self.labels[idx]

def load_data(config):
    """Load and analyze training data"""
    with open(config.TRAIN_LABELS_JSON, 'r', encoding='utf-8') as f:
        labels_dict = json.load(f)
    
    image_paths = []
    labels = []
    
    for filename, label_str in labels_dict.items():
        img_path = Path(config.TRAIN_IMG_DIR) / filename
        if img_path.exists():
            image_paths.append(str(img_path))
            labels.append(config.LABEL_MAPPING[label_str])
    
    print(f"✅ Loaded {len(image_paths)} training images")
    
    # Detailed class analysis
    label_counts = Counter(labels)
    print("\n📊 Detailed class distribution:")
    total = len(labels)
    
    for label_idx, count in sorted(label_counts.items()):
        label_name = config.INDEX_TO_LABEL[label_idx]
        percentage = 100.0 * count / total
        print(f"  {label_name}: {count} images ({percentage:.1f}%)")
    
    # Calculate imbalance ratio
    max_count = max(label_counts.values())
    min_count = min(label_counts.values())
    imbalance_ratio = max_count / min_count
    print(f"\n⚖️ Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    if imbalance_ratio > 3:
        print("⚠️ High class imbalance detected - using advanced rebalancing")
    
    return image_paths, labels

def main():
    """Main ultra training function"""
    print("="*60)
    print("🎯 ENT CLASSIFICATION - ULTRA HIGH ACCURACY VERSION")
    print("🚀 Target: 96%+ Accuracy")
    print("="*60)
    
    config = UltraConfig()
    trainer = UltraTrainer(config)
    
    # Load data
    image_paths, labels = load_data(config)
    
    # Train ensemble
    if config.USE_ENSEMBLE:
        models, accuracy = trainer.train_ensemble(image_paths, labels)
        print(f"\n🏆 Ensemble training completed! Best accuracy: {accuracy:.2f}%")
    else:
        model, accuracy = trainer.train_single_model(image_paths, labels, config.ENSEMBLE_MODELS[0])
        print(f"\n🏆 Single model training completed! Best accuracy: {accuracy:.2f}%")
    
    if accuracy >= 96.0:
        print("🎉 TARGET ACHIEVED! 96%+ accuracy reached!")
    else:
        print(f"🔄 Current best: {accuracy:.2f}% - Continue training or tune hyperparameters")

if __name__ == "__main__":
    main() 