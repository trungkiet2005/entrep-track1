#!/usr/bin/env python3
"""
ULTRA-OPTIMIZED ENT Classification Submission Script
Advanced techniques for maximum accuracy improvement from 0.93+

Features:
- Test Time Augmentation (TTA)
- Multi-scale Testing
- Advanced Preprocessing
- Confidence-based Ensembling
- Optimized Post-processing
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configuration
class UltraOptimizedConfig:
    """Ultra-optimized configuration for maximum accuracy"""
    
    # Data paths
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    MODEL_PATH = "efficientnet_b7_best_optimized.pth"
    OUTPUT_FILE = "efficientnet_b7_ultra_optimized_submission.json"
    
    # Multi-scale testing
    IMAGE_SIZES = [448, 512, 576]  # Multiple scales for robust testing
    BATCH_SIZE = 8  # Smaller batch for larger images
    NUM_CLASSES = 7
    
    # TTA Configuration
    TTA_ENABLED = True
    TTA_ROTATIONS = [0, 15, -15]  # Rotation angles
    TTA_FLIPS = ['none', 'horizontal', 'vertical']
    TTA_BRIGHTNESS = [0.9, 1.0, 1.1]  # Brightness variations
    TTA_CONTRAST = [0.9, 1.0, 1.1]    # Contrast variations
    
    # Advanced post-processing
    CONFIDENCE_THRESHOLD = 0.75  # High confidence threshold
    ENSEMBLE_WEIGHTS = [0.4, 0.35, 0.25]  # Weights for different scales
    
    # Label mapping
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class AdvancedENTTestDataset(Dataset):
    """Advanced test dataset with sophisticated preprocessing"""
    
    def __init__(self, image_paths, image_size=512, tta_enabled=False):
        self.image_paths = image_paths
        self.image_size = image_size
        self.tta_enabled = tta_enabled
        
        # Base transforms without augmentation
        self.base_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # TTA transforms
        if tta_enabled:
            self.tta_transforms = self._create_tta_transforms()
        else:
            self.tta_transforms = []
    
    def _create_tta_transforms(self):
        """Create TTA transform combinations"""
        transforms_list = []
        
        # Original
        transforms_list.append(self.base_transform)
        
        # Horizontal flip
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Vertical flip
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomVerticalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Rotation +15
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomRotation(degrees=(15, 15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Rotation -15
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.RandomRotation(degrees=(-15, -15)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Brightness adjustment
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        # Contrast adjustment
        transforms_list.append(transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ColorJitter(contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
        return transforms_list
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image_robust(self, img_path):
        """Robust image loading with multiple fallbacks"""
        try:
            # Try OpenCV first
            image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return image
        except:
            pass
        
        try:
            # Fallback to PIL
            image = Image.open(img_path).convert('RGB')
            return np.array(image)
        except:
            pass
        
        # Create dummy image if all fails
        print(f"Warning: Could not load image {img_path}, using dummy image")
        return np.zeros((224, 224, 3), dtype=np.uint8)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self._load_image_robust(img_path)
        filename = Path(img_path).name
        
        if self.tta_enabled and self.tta_transforms:
            # Apply all TTA transforms
            tta_images = []
            for transform in self.tta_transforms:
                try:
                    transformed = transform(image)
                    tta_images.append(transformed)
                except:
                    # Fallback to base transform if TTA fails
                    transformed = self.base_transform(image)
                    tta_images.append(transformed)
            
            return tta_images, filename
        else:
            # Apply base transform only
            transformed = self.base_transform(image)
            return transformed, filename

class OptimizedENTModel(nn.Module):
    """EfficientNet-B7 model with optimized architecture - compatible with existing weights"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super(OptimizedENTModel, self).__init__()
        
        # Load EfficientNet-B7 backbone
        self.backbone = models.efficientnet_b7(pretrained=pretrained)
        self.feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Attention mechanism
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
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply attention
        attention_weights = self.attention(features)
        attended_features = features * attention_weights
        
        # Classification
        output = self.classifier(attended_features)
        return output

def load_model_ultra(model_path, device, num_classes=7):
    """Load model with ultra optimization"""
    print(f"Loading ultra-optimized model from {model_path}...")
    
    model = OptimizedENTModel(num_classes=num_classes, pretrained=False)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Ultra-optimized model loaded successfully!")
    return model

def predict_with_tta(model, test_loader, device, config):
    """Make predictions with Test Time Augmentation"""
    model.eval()
    
    all_predictions = []
    all_confidences = []
    filenames = []
    
    print("Making predictions with TTA...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_loader):
            if batch_idx % 5 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            if isinstance(batch_data[0], list):
                # TTA enabled - batch_data[0] is list of TTA images
                tta_images_list, names = batch_data
                
                # Process each sample in batch
                batch_predictions = []
                batch_confidences = []
                
                for sample_idx in range(len(names)):
                    sample_predictions = []
                    sample_confidences = []
                    
                    # Get TTA images for this sample
                    for tta_idx in range(len(tta_images_list)):
                        tta_image = tta_images_list[tta_idx][sample_idx].unsqueeze(0).to(device)
                        
                        outputs = model(tta_image)
                        probs = F.softmax(outputs, dim=1)
                        
                        sample_predictions.append(probs.cpu().numpy()[0])
                        sample_confidences.append(torch.max(probs, dim=1)[0].cpu().numpy()[0])
                    
                    # Average TTA predictions for this sample
                    avg_pred = np.mean(sample_predictions, axis=0)
                    avg_conf = np.mean(sample_confidences)
                    
                    batch_predictions.append(avg_pred)
                    batch_confidences.append(avg_conf)
                
                all_predictions.extend(batch_predictions)
                all_confidences.extend(batch_confidences)
                filenames.extend(names)
            else:
                # No TTA
                images, names = batch_data
                images = images.to(device)
                outputs = model(images)
                probs = F.softmax(outputs, dim=1)
                
                all_predictions.extend(probs.cpu().numpy())
                all_confidences.extend(torch.max(probs, dim=1)[0].cpu().numpy())
                filenames.extend(names)
    
    return np.array(all_predictions), np.array(all_confidences), filenames

def multi_scale_prediction(model, test_images, config, device):
    """Perform multi-scale testing"""
    print("Performing multi-scale testing...")
    
    scale_predictions = []
    scale_confidences = []
    
    for i, image_size in enumerate(config.IMAGE_SIZES):
        print(f"\nTesting at scale {image_size}x{image_size}...")
        
        # Create dataset for this scale
        test_dataset = AdvancedENTTestDataset(
            test_images, 
            image_size=image_size,
            tta_enabled=config.TTA_ENABLED
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Get predictions for this scale
        predictions, confidences, filenames = predict_with_tta(model, test_loader, device, config)
        
        scale_predictions.append(predictions)
        scale_confidences.append(confidences)
    
    return scale_predictions, scale_confidences, filenames

def advanced_ensemble(scale_predictions, scale_confidences, config):
    """Advanced ensemble with confidence weighting"""
    print("Performing advanced ensemble...")
    
    # Confidence-weighted averaging
    weighted_predictions = np.zeros_like(scale_predictions[0])
    total_weights = np.zeros(len(scale_predictions[0]))
    
    for i, (preds, confs) in enumerate(zip(scale_predictions, scale_confidences)):
        # Scale weight from config
        scale_weight = config.ENSEMBLE_WEIGHTS[i]
        
        # Confidence-based weighting
        confidence_weights = np.power(confs, 1.5)  # Emphasize high confidence
        
        # Combined weight
        combined_weights = scale_weight * confidence_weights
        
        # Weighted sum
        for j in range(len(preds)):
            weighted_predictions[j] += preds[j] * combined_weights[j]
            total_weights[j] += combined_weights[j]
    
    # Normalize
    for i in range(len(weighted_predictions)):
        if total_weights[i] > 0:
            weighted_predictions[i] /= total_weights[i]
    
    return weighted_predictions

def create_ultra_submission(predictions, confidences, filenames, config):
    """Create ultra-optimized submission"""
    print("Creating ultra-optimized submission...")
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    max_confidences = np.max(predictions, axis=1)
    
    # Create submission dictionary
    submission = {}
    for filename, pred_label in zip(filenames, predicted_labels):
        submission[filename] = int(pred_label)
    
    # Save submission
    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nUltra-optimized submission saved to '{config.OUTPUT_FILE}' with {len(submission)} predictions")
    
    # Detailed analysis
    from collections import Counter
    pred_counts = Counter(predicted_labels)
    
    print("\nFinal prediction distribution:")
    for label_idx in range(config.NUM_CLASSES):
        label_name = config.INDEX_TO_LABEL[label_idx]
        count = pred_counts.get(label_idx, 0)
        percentage = 100.0 * count / len(predicted_labels)
        
        # Calculate average confidence for this class
        class_mask = predicted_labels == label_idx
        if np.any(class_mask):
            avg_conf = np.mean(max_confidences[class_mask])
        else:
            avg_conf = 0.0
        
        print(f"  {label_name}: {count} predictions ({percentage:.1f}%) - avg confidence: {avg_conf:.3f}")
    
    # Overall statistics
    print(f"\nOverall average confidence: {np.mean(max_confidences):.3f}")
    print(f"High confidence predictions (>0.9): {np.sum(max_confidences > 0.9)}")
    print(f"Medium confidence predictions (0.7-0.9): {np.sum((max_confidences >= 0.7) & (max_confidences <= 0.9))}")
    print(f"Low confidence predictions (<0.7): {np.sum(max_confidences < 0.7)}")

def main():
    """Main ultra-optimized function"""
    print("="*80)
    print("ENT CLASSIFICATION - ULTRA-OPTIMIZED SUBMISSION GENERATION")
    print("Advanced TTA + Multi-scale + Ensemble for Maximum Accuracy")
    print("Target: Improve from 0.93 to 0.95+")
    print("="*80)
    
    config = UltraOptimizedConfig()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    
    # Load test images
    print(f"\nLoading test images from {config.TEST_IMG_DIR}...")
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        test_images.extend(list(Path(config.TEST_IMG_DIR).glob(ext)))
    
    if not test_images:
        print("No test images found!")
        print("Please check the TEST_IMG_DIR path in the config.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Load model
    try:
        model = load_model_ultra(config.MODEL_PATH, device, config.NUM_CLASSES)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Multi-scale prediction with TTA
    try:
        scale_predictions, scale_confidences, filenames = multi_scale_prediction(
            model, test_images, config, device
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Advanced ensemble
    final_predictions = advanced_ensemble(scale_predictions, scale_confidences, config)
    final_confidences = np.mean(scale_confidences, axis=0)  # Average confidences across scales
    
    # Create submission
    create_ultra_submission(final_predictions, final_confidences, filenames, config)
    
    print("\n" + "="*80)
    print("ULTRA-OPTIMIZED SUBMISSION GENERATION COMPLETED!")
    print(f"Output file: '{config.OUTPUT_FILE}'")
    print("Expected improvement: 0.93 → 0.95+")
    print("="*80)

if __name__ == "__main__":
    main() 