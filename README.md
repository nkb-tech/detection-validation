# Detection Validation Framework

A comprehensive framework for validating object detection models, with a focus on YOLO models. This framework provides tools for converting YOLO format datasets to CSV, running inference with YOLO models, and calculating detailed evaluation metrics.

## Features

- Convert YOLO format datasets to CSV format
- Run inference with YOLO models on image datasets
- Calculate comprehensive evaluation metrics:
  - mAP (mean Average Precision) at IoU 0.5 and 0.5:0.95
  - Precision-Recall curves
  - Per-class metrics (AP, Precision, Recall, F1)
  - Confidence threshold analysis
- Generate detailed evaluation reports and visualizations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Convert YOLO Dataset to CSV

Convert your YOLO format dataset to CSV format for evaluation:

```bash
python data/yolo2csv.py \
    --images_list path/to/images.txt \
    --output output.csv \
    --class_names path/to/classes.txt
```

### 2. Run Model Inference

Run inference with your YOLO model:

```bash
python inference/yolo_infer.py \
    --weights path/to/model.pt \
    --txt_path path/to/images.txt \
    --output_dir results \
    --device 0 \
    --cfg path/to/config.yaml
```

### 3. Evaluate Results

Calculate evaluation metrics:

```bash
python validation/val.py \
    --gt path/to/ground_truth.csv \
    --pred path/to/predictions.csv \
    --output_dir results \
    --save_csv True \
    --conf_thresholds "0.1,0.25,0.5,0.75,0.9"
```

## Output

The evaluation script generates:
- Detailed per-class metrics in CSV format
- Precision-Recall curves visualization
- Overall metrics including mAP, precision, recall, and F1 scores
- Confidence threshold analysis

## Requirements

- Python 3.7+
- PyTorch
- OpenCV
- Pandas
- NumPy
- Matplotlib
- Ultralytics YOLO
