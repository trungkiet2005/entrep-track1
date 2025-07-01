#!/usr/bin/env python3
"""
Script to create submission using pre-trained EfficientNet-B7 model
Uses the trained efficientnet_b7_best_optimized.pth model for ENT classification
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

# Configuration
class SubmissionConfig:
    """Configuration for submission generation"""
    
    # Data paths
    TEST_IMG_DIR = "D:/PublicData/PublicTest/"
    MODEL_PATH = "efficientnet_b7_best_optimized.pth"
    OUTPUT_FILE = "efficientnet_b7_submission.json"
    
    # Model parameters (must match training config)
    IMAGE_SIZE = 512
    BATCH_SIZE = 16  # Can be larger for inference
    NUM_CLASSES = 7
    
    # Label mapping (must match training)
    LABEL_MAPPING = {
        'nose-right': 0, 'nose-left': 1, 'ear-right': 2, 'ear-left': 3,
        'vc-open': 4, 'vc-closed': 5, 'throat': 6
    }
    
    INDEX_TO_LABEL = {v: k for k, v in LABEL_MAPPING.items()}

class ENTTestDataset(Dataset):
    """Test dataset for ENT images"""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Load image
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
        
        return image, Path(img_path).name

class OptimizedENTModel(nn.Module):
    """EfficientNet-B7 model with optimized architecture"""
    
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

def get_test_transforms(image_size=512):
    """Test transforms without augmentation"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def load_model(model_path, device, num_classes=7):
    """Load pre-trained model"""
    print(f"Loading model from {model_path}...")
    
    # Create model
    model = OptimizedENTModel(num_classes=num_classes, pretrained=False)
    
    # Load weights
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
    
    print("Model loaded successfully!")
    return model

def predict_test_set(model, test_loader, device):
    """Make predictions on test set"""
    model.eval()
    
    all_predictions = []
    filenames = []
    
    print("Making predictions...")
    with torch.no_grad():
        for batch_idx, (images, names) in enumerate(test_loader):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
            
            images = images.to(device)
            outputs = model(images)
            predictions = F.softmax(outputs, dim=1).cpu().numpy()
            
            all_predictions.append(predictions)
            filenames.extend(names)
    
    return np.vstack(all_predictions), filenames

def create_submission(predictions, filenames, config):
    """Create submission file"""
    predicted_labels = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)
    
    submission = {}
    for filename, pred_label in zip(filenames, predicted_labels):
        submission[filename] = int(pred_label)
    
    # Save submission
    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(submission, f, indent=2)
    
    print(f"\nSubmission saved to '{config.OUTPUT_FILE}' with {len(submission)} predictions")
    
    # Analyze predictions
    from collections import Counter
    pred_counts = Counter(predicted_labels)
    print("\nPrediction distribution:")
    for label_idx in range(config.NUM_CLASSES):
        label_name = config.INDEX_TO_LABEL[label_idx]
        count = pred_counts.get(label_idx, 0)
        percentage = 100.0 * count / len(predicted_labels)
        avg_conf = np.mean([conf for pred, conf in zip(predicted_labels, confidence_scores) if pred == label_idx])
        print(f"  {label_name}: {count} predictions ({percentage:.1f}%) - avg confidence: {avg_conf:.3f}")
    
    # Overall confidence
    print(f"\nOverall average confidence: {np.mean(confidence_scores):.3f}")
    print(f"Low confidence predictions (<0.7): {np.sum(confidence_scores < 0.7)}")

def main():
    """Main function"""
    print("="*60)
    print("ENT CLASSIFICATION - SUBMISSION GENERATION")
    print("Using EfficientNet-B7 Pre-trained Model")
    print("="*60)
    
    config = SubmissionConfig()
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test images
    print(f"\nLoading test images from {config.TEST_IMG_DIR}...")
    test_images = list(Path(config.TEST_IMG_DIR).glob("*.png"))
    test_images.extend(list(Path(config.TEST_IMG_DIR).glob("*.jpg")))
    test_images.extend(list(Path(config.TEST_IMG_DIR).glob("*.jpeg")))
    
    if not test_images:
        print("No test images found!")
        print("Please check the TEST_IMG_DIR path in the config.")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Create test dataset and loader
    test_transforms = get_test_transforms(config.IMAGE_SIZE)
    test_dataset = ENTTestDataset(test_images, transform=test_transforms)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load model
    try:
        model = load_model(config.MODEL_PATH, device, config.NUM_CLASSES)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    try:
        predictions, filenames = predict_test_set(model, test_loader, device)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Create submission
    create_submission(predictions, filenames, config)
    
    print("\n" + "="*60)
    print("SUBMISSION GENERATION COMPLETED!")
    print(f"Output file: '{config.OUTPUT_FILE}'")
    print("="*60)

if __name__ == "__main__":
    main() 