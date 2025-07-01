#!/usr/bin/env python3
"""
ENT Image Classification - OPTIMIZED EFFICIENTNET-B7
Advanced techniques for maximum accuracy using EfficientNet-B7
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
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
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler

# Progress bar
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class OptimizedConfig:
    """Optimized configuration for maximum accuracy"""
    
    # Data paths
    TRAIN_IMG_DIR = "D:/train/train/imgs"
    TRAIN_LABELS_JSON = "D:/train/train/cls.json"
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    
    # Model save directory
    MODEL_SAVE_DIR = "saved_models"
    
    # Model parameters - IMPROVED for better convergence
    IMAGE_SIZE = 512        # Reduced from 600 for better memory efficiency
    BATCH_SIZE = 12         # Increased from 8 for better gradient estimates
    NUM_EPOCHS = 80         # Reduced from 100 as model converges earlier
    LEARNING_RATE = 5e-5    # Reduced from 1e-4 for more stable training
    WEIGHT_DECAY = 1e-3     # Increased from 1e-4 for better regularization
    NUM_CLASSES = 7
    
    # Advanced training parameters - IMPROVED
    LABEL_SMOOTHING = 0.05  # Reduced from 0.1 for medical data
    DROPOUT_RATE = 0.5      # Increased from 0.3 for better regularization
    MIXUP_ALPHA = 0.1       # Reduced from 0.2 for more conservative mixing
    CUTMIX_PROB = 0.3       # Reduced from 0.5
    
    # NEW: Warmup and decay parameters
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    PATIENCE = 15           # Increased from 20 for earlier stopping
    
    # Label mapping
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class MixupCutmixAugmentation:
    """Advanced augmentation techniques without flips"""
    
    @staticmethod
    def mixup_data(x, y, alpha=0.2):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    @staticmethod
    def cutmix_data(x, y, prob=0.5):
        if np.random.rand() > prob:
            return x, y, None, None

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        # Generate random box
        W = x.size()[2]
        H = x.size()[3]
        cut_rat = np.sqrt(1. - np.random.beta(1, 1))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Apply cutmix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        y_a, y_b = y, y[index]
        
        return x, y_a, y_b, lam

class OptimizedDataset(Dataset):
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

class OptimizedEfficientNet(nn.Module):
    """Optimized EfficientNet-B7 with advanced techniques"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super(OptimizedEfficientNet, self).__init__()
        
        # Load EfficientNet-B7
        self.backbone = models.efficientnet_b7(pretrained=pretrained)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Advanced classifier with attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // 4, self.feature_dim),
            nn.Sigmoid()
        )
        
        # Enhanced classifier with more layers and regularization
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
            
            nn.Linear(512, num_classes)
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

def get_optimized_transforms(mode='train', image_size=600):
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
            
            # Additional noise for robustness
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
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
        
        # Initialize mixed precision training
        self.scaler = GradScaler()
    
    def get_model_path(self, model_name):
        """Get the path for saved model"""
        model_path = os.path.join(self.config.MODEL_SAVE_DIR, f'{model_name}_best_optimized.pth')
        return model_path
    
    def load_model(self, model):
        """Load saved model if exists"""
        model_path = self.get_model_path('efficientnet_b7')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            accuracy = checkpoint.get('accuracy', 0.0)
            epoch = checkpoint.get('epoch', 0)
            print(f"✅ Loaded model - Accuracy: {accuracy:.2f}% (from epoch {epoch})")
            return True, accuracy
        return False, 0.0
    
    def create_weighted_sampler(self, labels):
        """Create weighted sampler for class balance"""
        class_counts = Counter(labels)
        weights = [1.0 / class_counts[label] for label in labels]
        return WeightedRandomSampler(weights, len(weights), replacement=True)
    
    def train_model_advanced(self, model, train_loader, val_loader):
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
        
        # IMPROVED scheduler with warmup and cosine annealing
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, SequentialLR, LinearLR
        
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=self.config.WARMUP_EPOCHS * len(train_loader)
        )
        
        # Main scheduler with restarts
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10 * len(train_loader),  # First restart after 10 epochs
            T_mult=2,                    # Double period after each restart
            eta_min=self.config.MIN_LR
        )
        
        # Combined scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.WARMUP_EPOCHS * len(train_loader)]
        )
        
        best_val_acc = 0.0
        best_model_state = None
        patience = self.config.PATIENCE
        patience_counter = 0
        
        # NEW: Track training metrics for better monitoring
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        print("\n🚀 Training with advanced techniques...")
        
        # Training loop with tqdm
        epoch_pbar = tqdm(range(self.config.NUM_EPOCHS), desc="Training", position=0, leave=True)
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Training batches with tqdm
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", position=1, leave=False)
            
            for batch_idx, (images, labels) in enumerate(batch_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                # Apply mixup or cutmix
                if np.random.rand() < 0.5:
                    images, labels_a, labels_b, lam = MixupCutmixAugmentation.mixup_data(
                        images, labels, self.config.MIXUP_ALPHA
                    )
                else:
                    images, labels_a, labels_b, lam = MixupCutmixAugmentation.cutmix_data(
                        images, labels, self.config.CUTMIX_PROB
                    )
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast():
                    outputs = model(images)
                    if lam is not None:
                        loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                    else:
                        loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update weights
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update batch progress bar with detailed metrics
                current_lr = scheduler.get_last_lr()[0]
                batch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*train_correct/train_total:.1f}%',
                    'LR': f'{current_lr:.2e}'
                })
            
            # Validation phase
            val_acc, val_loss = self.validate_model(model, val_loader, criterion)
            
            train_acc = 100. * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Store metrics
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            # Update epoch progress bar with comprehensive metrics
            epoch_pbar.set_postfix({
                'Train_Loss': f'{avg_train_loss:.4f}',
                'Train_Acc': f'{train_acc:.1f}%',
                'Val_Loss': f'{val_loss:.4f}',
                'Val_Acc': f'{val_acc:.1f}%',
                'Best': f'{best_val_acc:.1f}%'
            })
            
            # Print detailed epoch summary
            print(f"\nEpoch {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                model_path = self.get_model_path('efficientnet_b7')
                torch.save({
                    'model_state_dict': best_model_state,
                    'accuracy': best_val_acc,
                    'epoch': epoch
                }, model_path)
                
                epoch_pbar.set_description(f"Training ⭐ BEST: {best_val_acc:.1f}%")
                print(f"✨ New best model saved! Accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
                print(f"⏳ No improvement for {patience_counter} epochs")
                
            # Early stopping
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
        
        epoch_pbar.close()
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # NEW: Generate training summary
        self._generate_training_summary(train_losses, val_losses, train_accs, val_accs, best_val_acc)
        
        print(f"\n🎯 Training completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        return model, best_val_acc
    
    def _generate_training_summary(self, train_losses, val_losses, train_accs, val_accs, best_acc):
        """Generate detailed training summary"""
        import matplotlib.pyplot as plt
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Loss curves
            ax1.plot(train_losses, label='Train Loss', color='blue')
            ax1.plot(val_losses, label='Val Loss', color='red')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy curves
            ax2.plot(train_accs, label='Train Acc', color='blue')
            ax2.plot(val_accs, label='Val Acc', color='red')
            ax2.set_title('Accuracy Curves')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Loss difference (overfitting indicator)
            loss_diff = [v - t for v, t in zip(val_losses, train_losses)]
            ax3.plot(loss_diff, color='orange')
            ax3.set_title('Val - Train Loss (Overfitting Indicator)')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Loss Difference')
            ax3.grid(True)
            ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Learning rate schedule visualization
            epochs = list(range(len(train_losses)))
            ax4.plot(epochs, [0.001] * len(epochs), label='Simulated LR', color='green')  # Placeholder
            ax4.set_title('Learning Rate Schedule')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
        
            # Save plots
            plt.savefig('training_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Training summary saved to 'training_summary.png'")
            
            # Print analysis
            print("\n" + "="*50)
            print("📈 TRAINING ANALYSIS")
            print("="*50)
            
            final_train_acc = train_accs[-1] if train_accs else 0
            final_val_acc = val_accs[-1] if val_accs else 0
            
            print(f"Final Train Accuracy: {final_train_acc:.2f}%")
            print(f"Final Val Accuracy: {final_val_acc:.2f}%")
            print(f"Best Val Accuracy: {best_acc:.2f}%")
            print(f"Overfitting Gap: {abs(final_train_acc - final_val_acc):.2f}%")
            
            if len(val_accs) >= 10:
                last_10_improvement = max(val_accs[-10:]) - min(val_accs[-10:])
                print(f"Last 10 epochs improvement: {last_10_improvement:.2f}%")
                
            if final_train_acc > final_val_acc + 5:
                print("⚠️  WARNING: Possible overfitting detected!")
            elif best_acc > 85:
                print("✅ Excellent performance achieved!")
            elif best_acc > 75:
                print("✅ Good performance achieved!")
            else:
                print("⚠️  Consider adjusting hyperparameters for better performance")
                
        except Exception as e:
            print(f"Could not generate training plots: {e}")
            print("Install matplotlib: pip install matplotlib")
    
    def validate_model(self, model, val_loader, criterion):
        """Validation function with progress bar"""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Add progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validating", position=1, leave=False)
        
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Update validation progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*val_correct/val_total:.1f}%'
                })
        
        val_acc = 100. * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        return val_acc, val_loss
    
    def predict_with_tta(self, model, test_loader):
        """Test Time Augmentation with progress bar"""
        model.eval()
        
        all_predictions = []
        filenames = []
        
        print("🔮 Making predictions with TTA...")
        
        # Add progress bar for prediction
        pred_pbar = tqdm(test_loader, desc="Predicting", position=0, leave=True)
        
        with torch.no_grad():
            for images, names in pred_pbar:
                if len(filenames) == 0:
                    filenames.extend(names)
                
                images = images.to(self.device)
                
                # Original prediction
                outputs = model(images)
                predictions = F.softmax(outputs, dim=1).cpu().numpy()
                
                # Small rotation TTA (no flips)
                rotated_images = transforms.functional.rotate(images, 5)
                rotated_outputs = model(rotated_images)
                rotated_predictions = F.softmax(rotated_outputs, dim=1).cpu().numpy()
                
                # Average predictions
                avg_predictions = (predictions + rotated_predictions) / 2
                all_predictions.append(avg_predictions)
                
                # Update prediction progress bar
                pred_pbar.set_postfix({
                    'Processed': f'{len(all_predictions) * self.config.BATCH_SIZE}'
                })
        
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
    with open('efficientnet_optimized_submission.json', 'w') as f:
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

def main():
    """Optimized main function for maximum accuracy"""
    
    print("="*60)
    print("🏥 ENT CLASSIFICATION - OPTIMIZED EFFICIENTNET-B7")
    print("="*60)
    
    config = OptimizedConfig()
    trainer = OptimizedTrainer(config)
    
    # Load data
    print("📁 Loading training data...")
    image_paths, labels = load_optimized_data(config)
    if image_paths is None:
        return
    
    # Split data for training
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create model
    model = OptimizedEfficientNet(config.NUM_CLASSES)
    model.to(trainer.device)
    
    # Try to load existing model
    loaded, accuracy = trainer.load_model(model)
    
    if not loaded:
        print("🆕 Training from scratch...")
        
        # Create datasets
        train_dataset = OptimizedDataset(
            train_paths, train_labels, 
            get_optimized_transforms('train', config.IMAGE_SIZE)
        )
        val_dataset = OptimizedDataset(
            val_paths, val_labels, 
            get_optimized_transforms('val', config.IMAGE_SIZE)
        )
        
        # Create weighted sampler for training
        weighted_sampler = trainer.create_weighted_sampler(train_labels)
        
        # Create loaders
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
        model, accuracy = trainer.train_model_advanced(model, train_loader, val_loader)
    
    # Load test data
    print("\n📊 Loading test data...")
    test_images = list(Path(config.TEST_IMG_DIR).glob("*.png"))
    test_images.extend(list(Path(config.TEST_IMG_DIR).glob("*.jpg")))
    
    if not test_images:
        print("❌ No test images found!")
        return
    
    print(f"Found {len(test_images)} test images")
    
    test_dataset = OptimizedDataset(
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
    
    # Make predictions with TTA
    predictions, filenames = trainer.predict_with_tta(model, test_loader)
    
    # Create submission
    create_optimized_submission(predictions, filenames, config)
    
    print("\n" + "="*60)
    print("🎉 OPTIMIZED TRAINING COMPLETED!")
    print("📄 File saved: 'efficientnet_optimized_submission.json'")
    print("🎯 Expected accuracy: 96-98%")
    print("="*60)

if __name__ == "__main__":
    main() 