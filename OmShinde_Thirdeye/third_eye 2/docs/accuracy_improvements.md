# YOLO Accuracy Improvements for Third Eye Navigation

This document outlines the comprehensive accuracy improvements implemented for better object detection, especially for moving objects and diverse environments.

## Overview of Improvements

### 1. Stronger Model Support
- **High-Performance Profile**: Uses `yolov8s.pt` with `img_size=448`
- **Ultra-Performance Profile**: Uses `yolov8m.pt` with `img_size=512`
- **Balanced Profile**: Enhanced with `img_size=384` (up from 256)

### 2. Enhanced Object Detection Classes
Expanded from 24 to 73 COCO classes including:
- **Vehicles**: airplane, train, boat (in addition to existing car, bus, truck, etc.)
- **Animals**: bird, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Sports/Recreation**: skateboard, surfboard, sports ball, frisbee, skis
- **Household Items**: Complete kitchen and living room objects
- **Navigation Hazards**: All furniture and obstacles

### 3. Advanced Tracking Features
- **Frame-to-frame tracking**: Already enabled (`tracker: true`)
- **EMA smoothing**: Reduces detection jitter
- **Configurable tracking parameters**: Max age, hit thresholds, IoU settings

### 4. Fine-tuning Capabilities
- **Domain-specific training**: Custom dataset support
- **Navigation-biased classes**: Prioritize mobility-critical objects
- **Hard negative mining**: Improve detection of challenging scenarios

## Performance Profiles

### Balanced (Default)
```bash
# No environment variable needed - this is the default
python run_depth.py
```
- Model: `yolov8n.pt`
- Image Size: `384px`
- Target FPS: `25`
- Best for: General use, good performance/accuracy balance

### High-Performance
```bash
export THIRD_EYE_PROFILE=high-performance
python run_depth.py
```
- Model: `yolov8s.pt` (4x more parameters than nano)
- Image Size: `448px`
- Target FPS: `30`
- Max Detections: `30`
- Best for: Desktop/laptop with dedicated GPU

### Ultra-Performance
```bash
export THIRD_EYE_PROFILE=ultra-performance
python run_depth.py
```
- Model: `yolov8m.pt` (10x more parameters than nano)
- Image Size: `512px`
- Target FPS: `25`
- Max Detections: `50`
- Augmentation: Enabled
- Best for: High-end hardware, maximum accuracy needed

## Quick Setup

### 1. Download Required Models
```bash
python download_models.py
```
This downloads: `yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt`, `yolov8x.pt`

### 2. Use Enhancement Script
```bash
# Apply all improvements
python enhance_accuracy.py --all

# Or apply specific improvements
python enhance_accuracy.py --profile high-performance --enable-tracking --validate
```

### 3. Test Performance
```bash
python validate_performance.py
```

## Configuration Details

### YOLO Settings by Profile

| Setting | Balanced | High-Perf | Ultra-Perf |
|---------|----------|-----------|------------|
| Model | yolov8n.pt | yolov8s.pt | yolov8m.pt |
| Image Size | 384 | 448 | 512 |
| Confidence | 0.55 | 0.5 | 0.45 |
| Max Detections | 20 | 30 | 50 |
| Augmentation | false | false | true |

### Tracking Configuration
```json
{
  "tracking": {
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "track_high_thresh": 0.6,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.7
  }
}
```

## Fine-tuning for Domain-Specific Accuracy

### 1. Prepare Training Data
```bash
# Collect navigation-specific images
python datasets/scripts/data_collection.py --scenario navigation

# Preprocess and split dataset
python datasets/scripts/preprocess.py
python datasets/scripts/split_dataset.py
```

### 2. Run Fine-tuning
```bash
# Setup fine-tuning (creates run_finetuning.py)
python enhance_accuracy.py --setup-finetuning

# Run training
python run_finetuning.py
```

### 3. Evaluate Results
```bash
python datasets/scripts/evaluate_model.py --model-path third_eye_finetuned/navigation_model/weights/best.pt
```

## Performance Optimization Tips

### For Moving Objects
1. **Enable Tracking**: Reduces false negatives on fast-moving objects
2. **Lower Confidence**: Use `conf=0.4-0.5` for better recall
3. **Higher FPS**: Increase camera FPS to 30+ for smoother tracking
4. **Reduce Motion Blur**: Use faster shutter speed if possible

### For Crowded Scenes
1. **Increase Max Detections**: Set `max_det=50+`
2. **Use Stronger Model**: Switch to `yolov8m.pt` or `yolov8l.pt`
3. **Enable Augmentation**: Helps with varied lighting/angles
4. **Lower IoU Threshold**: Reduces over-suppression of nearby objects

### For Specific Environments
1. **Indoor Navigation**: Focus on furniture, doorways, stairs
2. **Outdoor Navigation**: Emphasize vehicles, traffic signs, curbs
3. **Crowded Areas**: Prioritize people, moving objects, obstacles

## Monitoring and Validation

### Performance Metrics
- **FPS**: Target 20-30 FPS depending on profile
- **Detection Count**: Average 5-15 objects per frame
- **Latency**: <50ms processing time per frame

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/health

# View performance logs
tail -f logs/third_eye.log
```

### A/B Testing
```bash
# Test different configurations
THIRD_EYE_PROFILE=balanced python validate_performance.py
THIRD_EYE_PROFILE=high-performance python validate_performance.py

# Compare results in performance_results.json
```

## Troubleshooting

### Low FPS
1. Reduce image size: `384 → 320 → 256`
2. Use smaller model: `yolov8m → yolov8s → yolov8n`
3. Increase frame skipping: `yolo_skip=2`
4. Reduce max detections: `max_det=15`

### Missing Detections
1. Lower confidence threshold: `conf=0.3`
2. Use stronger model: `yolov8s → yolov8m`
3. Increase image size: `384 → 448`
4. Enable augmentation for training data

### False Positives
1. Raise confidence threshold: `conf=0.6`
2. Adjust IoU threshold: `iou=0.6`
3. Fine-tune on domain-specific data
4. Use class-specific confidence thresholds

## Advanced Features

### Custom Class Weights
```python
# In run_depth.py, modify class confidence per category
class_weights = {
    'person': 1.2,      # Boost person detection
    'car': 1.1,         # Boost vehicle detection  
    'traffic light': 1.3 # Critical for navigation
}
```

### Ensemble Detection
```json
{
  "yolo": {
    "ensemble": true,
    "ensemble_models": ["yolov8n.pt", "yolov8s.pt"],
    "ensemble_weights": [0.4, 0.6]
  }
}
```

### Test-Time Augmentation
```json
{
  "yolo": {
    "augment": true,
    "tta_scales": [0.8, 1.0, 1.2],
    "tta_flips": [false, true]
  }
}
```

## Next Steps

1. **Collect Domain Data**: Gather navigation-specific images
2. **Fine-tune Models**: Train on your specific use cases
3. **Monitor Performance**: Track accuracy improvements over time
4. **Iterate**: Continuously improve based on real-world usage

For questions or issues, check the logs in `logs/third_eye.log` or run the health check endpoint.