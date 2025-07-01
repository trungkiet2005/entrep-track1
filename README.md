# ENT Image Classification Challenge - High Accuracy Solution

Giải pháp phân loại hình ảnh nội soi tai mũi họng với độ chính xác cao cho ENTRep Challenge.

## 🎯 Mô tả bài toán

Phân loại hình ảnh nội soi tai mũi họng thành 7 nhãn:
- 0: `nose-right` (325 images)
- 1: `nose-left` (290 images)  
- 2: `ear-right` (156 images)
- 3: `ear-left` (133 images)
- 4: `vc-open` (159 images)
- 5: `vc-closed` (147 images)
- 6: `throat` (81 images)

## 🚀 Các tính năng chính

### 1. Kiến trúc Model tiên tiến
- **Ensemble Learning**: Kết hợp ResNet101 và EfficientNet-B4
- **Transfer Learning**: Sử dụng pretrained models từ ImageNet
- **Enhanced Classifier**: Dropout, BatchNorm, và multi-layer classification head

### 2. Kỹ thuật Data Augmentation
- Random rotation, flipping, color jittering
- Resize về 384x384 để tối ưu hiệu suất
- Normalization theo ImageNet standards

### 3. Training Strategies
- **Label Smoothing**: Giảm overfitting
- **AdamW Optimizer**: Cải thiện generalization
- **Cosine Annealing LR**: Học động tối ưu
- **Early Stopping**: Tránh overfitting

### 4. Ensemble Prediction
- Voting từ nhiều models
- Softmax averaging cho dự đoán cuối cùng

## 📁 Cấu trúc Project

```
ENTRep/
├── ent_classifier.py      # Main classification code
├── requirements.txt       # Dependencies
├── run_ent_classification.py  # Quick runner script  
├── README.md             # Documentation
└── submission.json       # Output predictions
```

## 🛠 Cài đặt và Chạy

### Cách 1: Chạy tự động (Khuyến nghị)
```bash
python run_ent_classification.py
```

### Cách 2: Chạy thủ công
```bash
# Install dependencies
pip install -r requirements.txt

# Run classification
python ent_classifier.py
```

## 📊 Cấu hình Model

```python
class ENTConfig:
    IMAGE_SIZE = 384        # Optimal size for balance between quality and speed
    BATCH_SIZE = 16         # Fits most GPUs
    NUM_EPOCHS = 50         # Sufficient for convergence
    LEARNING_RATE = 1e-4    # Conservative learning rate
    WEIGHT_DECAY = 1e-4     # L2 regularization
    NUM_CLASSES = 7         # ENT anatomical regions
```

## 🎯 Chiến lược để đạt độ chính xác cao

### 1. Data Quality
- Resize thông minh từ 640x480 → 384x384
- Augmentation phù hợp với medical images
- Proper train/validation split với stratification

### 2. Model Architecture
- **ResNet101**: Mạng sâu với residual connections
- **EfficientNet-B4**: Efficient scaling của width/depth/resolution
- **Enhanced Classifier**: Multi-layer với regularization

### 3. Training Techniques
- **Label Smoothing (0.1)**: Giảm overconfidence
- **AdamW**: Weight decay tách biệt
- **Cosine Annealing**: Learning rate scheduling smooth

### 4. Ensemble Strategy
- Kết hợp predictions từ multiple models
- Softmax averaging thay vì hard voting
- Cross-validation cho robust evaluation

## 📈 Kết quả mong đợi

Với configuration hiện tại, solution này được thiết kế để đạt:
- **Validation Accuracy**: 85-95%
- **Robust Performance**: Ít overfitting
- **Fast Inference**: Optimized cho production

## 🔧 Tuning Parameters

Để cải thiện thêm performance:

```python
# Tăng model complexity
IMAGE_SIZE = 512           # Higher resolution
BATCH_SIZE = 8            # Smaller batch cho stability

# Advanced augmentation
RandomRotation(30)        # Stronger rotation
ColorJitter(0.3, 0.3)    # More color variation

# Training tweaks  
NUM_EPOCHS = 100          # Longer training
LEARNING_RATE = 5e-5      # Lower learning rate
```

## 📋 Đầu ra

File `submission.json` với format:
```json
{
  "image1.png": 0,
  "image2.png": 4,
  "image3.png": 6
}
```

## 🏥 Medical Image Considerations

Solution này được thiết kế đặc biệt cho medical imaging:
- Conservative augmentation (không làm biến dạng anatomy)
- Label smoothing (uncertain trong medical diagnosis)
- Ensemble approach (như multiple doctors consensus)
- High resolution preservation (chi tiết y tế quan trọng)

## 🎯 Tips để đạt kết quả tốt nhất

1. **Đảm bảo data paths đúng** trong `ENTConfig`
2. **Sử dụng GPU** cho training nhanh
3. **Monitor validation accuracy** để early stopping
4. **Ensemble nhiều runs** để stability cao hơn
5. **Fine-tune hyperparameters** dựa trên validation results

## 📧 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra data paths trong config
2. Đảm bảo đủ RAM/GPU memory
3. Verify image formats (PNG/JPG)
4. Check dependencies versions

---

**Good luck với ENTRep Challenge! 🏆** 