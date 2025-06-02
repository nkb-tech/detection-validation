"""
YOLO Model Inference

This script performs inference using a YOLO model on a list of images and saves the predictions
in CSV format for evaluation. It supports both single-class and multi-class detection.
"""

import cv2
import yaml
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

class Predictor:
    """
    YOLO model predictor for object detection.
    
    This class handles loading the model, running inference, and saving predictions
    in a format suitable for evaluation.
    """

    def __init__(self, weights, txt_path, output_dir, device=0, cfg=None):
        """
        Initialize the predictor.
        
        Args:
            weights (str): Path to the YOLO model weights (.pt file)
            txt_path (str): Path to text file containing list of image paths
            output_dir (str): Directory to save prediction results
            device (int/str): CUDA device ID or 'cpu'
            cfg (str, optional): Path to YAML configuration file
        """
        self.model = YOLO(weights)
        self.txt_path = txt_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device

        self.params = {}
        if cfg is not None:
            with open(cfg, 'r') as f:
                cfg = yaml.safe_load(f)
            assert cfg is not None, "Config is empty!"
            self.params.update(cfg)

    def predict(self, img):
        """
        Run inference on a single image.
        
        Args:
            img (numpy.ndarray): Input image in BGR format
            
        Returns:
            list: List of dictionaries containing predictions:
                - bbox: [xmin, ymin, xmax, ymax]
                - score: confidence score
                - class: class ID
        """
        preds = []
        results = self.model(img, device=self.device, **self.params, verbose=False)[0]
        for det in results.boxes:
            cls = int(det.cls.item())
            xyxy = det.xyxy.cpu().numpy()[0]
            xmin, ymin, xmax, ymax = xyxy
            bconf = float(det.conf.item())
            preds.append({'bbox':[xmin, ymin, xmax, ymax], 'score':bconf, 'class': cls})
        return preds

    def run_prediction(self):
        """
        Run inference on all images and save predictions to CSV.
        
        This method:
        1. Reads the list of images
        2. Runs inference on each image
        3. Collects predictions
        4. Saves results to CSV format
        """
        # Read list of images
        with open(self.txt_path) as f:
            img_paths = [line.strip() for line in f if line.strip()]

        predictions_df = defaultdict(list)

        # Run inference on each image
        for img_idx, img_path in enumerate(tqdm(img_paths, desc='Inference')):
            img8_3ch = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img8_3ch is None or len(img8_3ch.shape)<3 or img8_3ch.dtype!=np.uint8:
                print(f"SKIP: cannot read {img_path}")
                continue
                
            preds = self.predict(img8_3ch)
            for det_pred in preds:
                predictions_df["image_path"].append(str(img_path))
                predictions_df["xmin"].append(det_pred['bbox'][0])
                predictions_df["ymin"].append(det_pred['bbox'][1])
                predictions_df["xmax"].append(det_pred['bbox'][2])
                predictions_df["ymax"].append(det_pred['bbox'][3])
                predictions_df["conf"].append(det_pred['score'])
                predictions_df["detection_label"].append(int(det_pred['class']))

        predictions_df = pd.DataFrame(predictions_df)

        try:
            predictions_df.to_csv(self.output_dir / "predictions.csv", index=False)
        except Exception as e:
            print('ERROR', e)

        print(f"Predictions saved to {self.output_dir}")

def main():
    """Main function to run YOLO inference."""
    parser = argparse.ArgumentParser(description='Run inference with YOLO model')
    parser.add_argument('--weights', type=str, required=True, 
                       help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--txt_path', type=str, required=True, 
                       help='Path to file containing list of image paths')
    parser.add_argument('--output_dir', type=str, default='results_eval', 
                       help='Directory to save prediction results')
    parser.add_argument('--device', type=str, default='0', 
                       help='CUDA device ID or "cpu"')
    parser.add_argument('--cfg', type=str, default=None, 
                       help='Path to YAML configuration file')
    args = parser.parse_args()

    predictor = Predictor(
        weights=args.weights,
        txt_path=args.txt_path,
        output_dir=args.output_dir,
        device=args.device,
        cfg=args.cfg
    )
    predictor.run_prediction()

if __name__ == '__main__':
    main()