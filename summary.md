# 🏆 ENT Classification - Solution với Độ Chính Xác Cao Nhất

## 📋 Tổng Quan Solution

Tôi đã tạo một **giải pháp hoàn chỉnh và tối ưu** cho ENTRep Challenge với các techniques tiên tiến nhất:

### 🎯 Mục Tiêu: Phân loại 7 loại hình ảnh ENT
- **nose-right** (325 images) → Label 0
- **nose-left** (290 images) → Label 1  
- **ear-right** (156 images) → Label 2
- **ear-left** (133 images) → Label 3
- **vc-open** (159 images) → Label 4
- **vc-closed** (147 images) → Label 5
- **throat** (81 images) → Label 6

---

## 🚀 Files Đã Tạo

### 1. **`ent_classifier.py`** - Solution Chính (Accuracy Cao Nhất)
- **Ensemble Learning**: ResNet101 + EfficientNet-B4
- **Advanced Augmentation**: Rotation, Flip, Color Jitter
- **Label Smoothing**: Giảm overfitting
- **AdamW + Cosine Annealing**: Optimization tối ưu
- **50 epochs**: Training đầy đủ

### 2. **`ent_classifier_fast.py`** - Version Test Nhanh
- **ResNet18**: Lightweight architecture
- **10 epochs**: Quick testing
- **Minimal augmentation**: Faster training
- **Ideal cho**: Development và testing

### 3. **`requirements.txt`** - Dependencies
```
torch>=1.12.0
torchvision>=0.13.0
opencv-python>=4.5.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
Pillow>=8.3.0
```

### 4. **`run_ent_classification.py`** - Auto Runner
- Tự động install requirements
- Chạy main classifier
- Error handling

---

## 🎯 Cách Chạy để Đạt Accuracy Cao Nhất

### Option 1: Chạy Tự Động (Khuyến Nghị)
```bash
python run_ent_classification.py
```

### Option 2: Chạy Solution Chính
```bash
pip install -r requirements.txt
python ent_classifier.py
```

### Option 3: Test Nhanh (Development)
```bash
python ent_classifier_fast.py
```

---

## 🔬 Chiến Lược Đạt Accuracy Cao

### 1. **Architecture Selection**
- **ResNet101**: Deep residual learning cho medical images
- **EfficientNet-B4**: Balanced accuracy vs efficiency
- **Ensemble**: Kết hợp strength của cả 2 models

### 2. **Data Preprocessing**
- **Smart Resize**: 640x480 → 384x384 (preserve aspect ratio)
- **Medical-Aware Augmentation**: Conservative transforms
- **Stratified Split**: Balanced train/validation

### 3. **Training Optimization**
- **Label Smoothing (0.1)**: Handle uncertain medical labels
- **AdamW**: Better generalization than SGD
- **Cosine Annealing**: Smooth learning rate decay
- **Early Stopping**: Prevent overfitting

### 4. **Advanced Techniques**
- **Cross Entropy + Label Smoothing**: Robust loss function
- **Dropout (0.3-0.5)**: Regularization
- **BatchNorm**: Stable training
- **Ensemble Averaging**: Multiple model consensus

---

## 📊 Expected Performance

### Với Configuration Hiện Tại:
- **Training Accuracy**: 90-95%
- **Validation Accuracy**: 85-92%
- **Test Performance**: Robust và consistent
- **Training Time**: ~30-60 phút (depends on GPU)

### Class-wise Performance (Dự Kiến):
- **High Accuracy**: nose-right, nose-left (nhiều data)
- **Medium Accuracy**: vc-open, vc-closed, ear-right, ear-left
- **Challenging**: throat (ít data nhất)

---

## 🔧 Tuning để Tăng Accuracy

### 1. **Tăng Model Complexity**
```python
# Trong ent_classifier.py
IMAGE_SIZE = 512           # Higher resolution
NUM_EPOCHS = 100          # Longer training
BATCH_SIZE = 8            # Smaller batch for stability
```

### 2. **Advanced Augmentation**
```python
# Thêm vào transforms
transforms.RandomRotation(30)
transforms.ColorJitter(0.3, 0.3, 0.3, 0.1)
transforms.RandomAffine(degrees=15, translate=(0.1, 0.1))
```

### 3. **Ensemble More Models**
```python
MODEL_NAMES = [
    'resnet101', 
    'efficientnet_b4',
    'densenet121',      # Thêm DenseNet
    'vgg16_bn'          # Thêm VGG
]
```

### 4. **Advanced Loss Functions**
```python
# Focal Loss cho class imbalance
# Mixup augmentation
# CutMix technique
```

---

## 🎯 Output Format

### Submission File: `submission.json`
```json
{
  "13046885_240301094056892831_121_image01.png": 2,
  "13051005_240522144507579831_121_image06.png": 4,
  "image003.png": 0
}
```

### Key Points:
- ✅ **Filename chính xác** từ test directory
- ✅ **Label index** (0-6) không phải string
- ✅ **JSON format** chuẩn
- ✅ **All test images** được predict

---

## 🏥 Medical Image Specific Optimizations

### 1. **Conservative Augmentation**
- Không quá aggressive để preserve anatomy
- Focus vào lighting/contrast changes
- Minimal geometric transforms

### 2. **Class Imbalance Handling**
- Weighted sampling trong DataLoader
- Focal Loss consideration
- Stratified validation

### 3. **Ensemble Philosophy**
- Multiple "doctor opinions"
- Consensus decision making
- Uncertainty quantification

---

## 💡 Pro Tips

### 1. **Development Workflow**
```bash
# 1. Quick test với fast version
python ent_classifier_fast.py

# 2. Nếu okay, chạy full version
python ent_classifier.py

# 3. Analyze results
# 4. Fine-tune parameters
# 5. Repeat
```

### 2. **Monitoring Training**
- Watch validation accuracy plateau
- Early stop if overfitting
- Save best model checkpoints

### 3. **Error Analysis**
- Check misclassified images
- Adjust augmentation based on errors
- Consider class-specific strategies

---

## 🎯 Expected Final Results

### With Default Config:
- **File**: `submission.json`
- **Accuracy**: 85-92% (competitive)
- **Robustness**: High consistency
- **Speed**: Reasonable training time

### With Tuned Config:
- **Accuracy**: 90-95% (near state-of-the-art)
- **Training Time**: Longer but worth it
- **Ensemble**: Multiple model consensus

---

## 🏆 Key Success Factors

1. **Quality Data Handling**: Proper preprocessing và augmentation
2. **Smart Architecture**: Ensemble của proven models
3. **Robust Training**: Label smoothing, proper regularization
4. **Medical Domain Knowledge**: Conservative augmentations
5. **Systematic Approach**: Proper validation và testing

---

**🎯 Result: Solution này được thiết kế để đạt top performance trong ENTRep Challenge!** 