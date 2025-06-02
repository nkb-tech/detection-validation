import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import torch
from torchvision.ops import box_iou
import warnings
import argparse
import os

warnings.filterwarnings('ignore')

class ObjectDetectionEvaluator:
    def __init__(self, gt_csv_path, pred_csv_path):
        """
        Initialize the evaluator with ground truth and prediction CSV files.
        
        Args:
            gt_csv_path (str): Path to ground truth CSV file
            pred_csv_path (str): Path to predictions CSV file
        """
        self.gt_df = pd.read_csv(gt_csv_path)
        self.pred_df = pd.read_csv(pred_csv_path)
        
        gt_classes = set(self.gt_df['label'].unique()) if not self.gt_df.empty else set()
        pred_classes = set(self.pred_df['detection_label'].unique()) if not self.pred_df.empty else set()
        self.classes = sorted(list(gt_classes.union(pred_classes)))
        
        gt_images = set(self.gt_df['image_path'].unique()) if not self.gt_df.empty else set()
        pred_images = set(self.pred_df['image_path'].unique()) if not self.pred_df.empty else set()
        self.all_images = sorted(list(gt_images.union(pred_images)))
        
        print(f"Found {len(self.classes)} classes: {self.classes}")
        print(f"Found {len(self.all_images)} total images")
        print(f"Images with GT: {len(gt_images)}")
        print(f"Images with predictions: {len(pred_images)}")
    
    def calculate_iou(self, boxes1, boxes2):
        """
        Calculate IoU between two sets of boxes using torchvision.ops.box_iou
        
        Args:
            boxes1 (torch.Tensor): Ground truth boxes [N, 4] in format [xmin, ymin, xmax, ymax]
            boxes2 (torch.Tensor): Predicted boxes [M, 4] in format [xmin, ymin, xmax, ymax]
        
        Returns:
            torch.Tensor: IoU matrix [N, M]
        """
        return box_iou(boxes1, boxes2)
    
    def get_image_data(self, image_path, class_name):
        """
        Get ground truth and prediction data for a specific image and class.
        
        Args:
            image_path (str): Path to the image
            class_name (str): Name of the class
        
        Returns:
            tuple: (gt_boxes, pred_boxes, pred_scores)
        """
        gt_mask = (self.gt_df['image_path'] == image_path) & (self.gt_df['label'] == class_name)
        gt_data = self.gt_df[gt_mask]
        
        if len(gt_data) > 0:
            gt_boxes = torch.tensor(gt_data[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
        else:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32)
        
        pred_mask = (self.pred_df['image_path'] == image_path) & (self.pred_df['detection_label'] == class_name)
        pred_data = self.pred_df[pred_mask]
        
        if len(pred_data) > 0:
            pred_boxes = torch.tensor(pred_data[['xmin', 'ymin', 'xmax', 'ymax']].values, dtype=torch.float32)
            pred_scores = torch.tensor(pred_data['conf'].values, dtype=torch.float32)
        else:
            pred_boxes = torch.empty((0, 4), dtype=torch.float32)
            pred_scores = torch.empty((0,), dtype=torch.float32)
        
        return gt_boxes, pred_boxes, pred_scores
    
    def calculate_precision_recall_at_iou(self, iou_threshold=0.5, conf_threshold=0.0):
        """
        Calculate precision and recall for each class at a specific IoU threshold.
        Returns precision/recall at best F1 score threshold.
        
        Args:
            iou_threshold (float): IoU threshold for considering a detection as correct
            conf_threshold (float): Minimum confidence threshold for filtering predictions
        
        Returns:
            dict: Dictionary containing precision, recall, and other metrics for each class
        """
        class_metrics = {}
        
        for class_name in self.classes:
            # Collect all detections and ground truths for this class across all images
            all_detections = []
            total_gt = 0
            
            for image_path in self.all_images:
                gt_boxes, pred_boxes, pred_scores = self.get_image_data(image_path, class_name)
                
                total_gt += len(gt_boxes)
                
                # Filter predictions by confidence threshold
                valid_pred_mask = pred_scores >= conf_threshold
                pred_boxes = pred_boxes[valid_pred_mask]
                pred_scores = pred_scores[valid_pred_mask]
                
                if len(pred_boxes) == 0:
                    continue
                
                # Track which GT boxes have been matched
                gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
                
                # Sort predictions by confidence
                sorted_indices = torch.argsort(pred_scores, descending=True)
                
                for pred_idx in sorted_indices:
                    pred_box = pred_boxes[pred_idx:pred_idx+1]
                    pred_conf = pred_scores[pred_idx].item()
                    
                    is_correct = False
                    
                    if len(gt_boxes) > 0:
                        ious = self.calculate_iou(gt_boxes, pred_box).squeeze(1)
                        
                        max_iou, max_idx = torch.max(ious, dim=0)
                        
                        if max_iou >= iou_threshold and not gt_matched[max_idx]:
                            is_correct = True
                            gt_matched[max_idx] = True
                    
                    all_detections.append((pred_conf, is_correct))
            
            # Sort all detections by confidence
            all_detections.sort(key=lambda x: x[0], reverse=True)
            
            precisions = []
            recalls = []
            f1_scores = []
            confs = []
            
            tp = 0
            fp = 0
            
            for conf, is_correct in all_detections:
                if is_correct:
                    tp += 1
                else:
                    fp += 1
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
                confs.append(conf)
            
            # Find best F1 score and corresponding precision/recall
            if len(f1_scores) > 0:
                best_f1_idx = np.argmax(f1_scores)
                best_precision = precisions[best_f1_idx]
                best_recall = recalls[best_f1_idx]
                best_f1 = f1_scores[best_f1_idx]
                best_conf = confs[best_f1_idx]
                
                # Add initial point for PR curve
                precisions = [1.0] + precisions
                recalls = [0.0] + recalls
            else:
                best_precision = 0.0
                best_recall = 0.0
                best_f1 = 0.0
                precisions = [1.0, 0.0]
                recalls = [0.0, 0.0]
                best_conf = 0.0
            
            class_metrics[class_name] = {
                'precisions': precisions,
                'recalls': recalls,
                'best_precision': best_precision,
                'best_recall': best_recall,
                'best_f1': best_f1,
                'best_conf': best_conf,
                'total_gt': total_gt,
                'total_detections': len(all_detections)
            }
        
        return class_metrics
    
    def calculate_ap(self, precisions, recalls):
        """
        Calculate Average Precision using all-point interpolation (modern standard).
        
        Args:
            precisions (list): List of precision values
            recalls (list): List of recall values
        
        Returns:
            float: Average Precision
        """
        if len(precisions) == 0 or len(recalls) == 0:
            return 0.0
        
        precisions = np.array(precisions)
        recalls = np.array(recalls)
        
        sorted_indices = np.argsort(recalls)
        recalls = recalls[sorted_indices]
        precisions = precisions[sorted_indices]
        
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([1.0], precisions, [0.0]))
        
        for i in range(len(precisions) - 2, -1, -1):
            precisions[i] = max(precisions[i], precisions[i + 1])
        
        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        
        return ap
    
    def calculate_map(self, iou_thresholds):
        """
        Calculate mean Average Precision over multiple IoU thresholds.
        
        Args:
            iou_thresholds (list): List of IoU thresholds
        
        Returns:
            dict: Dictionary containing mAP and per-class AP values
        """
        all_aps = defaultdict(list) 
        
        for iou_thresh in iou_thresholds:
            class_metrics = self.calculate_precision_recall_at_iou(iou_thresh)
            
            for class_name, metrics in class_metrics.items():
                if metrics['total_gt'] > 0: 
                    ap = self.calculate_ap(metrics['precisions'], metrics['recalls'])
                    all_aps[class_name].append(ap)
                else:
                    all_aps[class_name].append(0.0)
        
        class_maps = {}
        for class_name, aps in all_aps.items():
            class_maps[class_name] = np.mean(aps)
        
        overall_map = np.mean(list(class_maps.values())) if class_maps else 0.0
        
        return {
            'mAP': overall_map,
            'class_mAPs': class_maps,
            'all_APs': dict(all_aps)
        }
    
    def plot_pr_curve(self, save_path='pr_curve.png'):
        """
        Plot and save Precision-Recall curves for all classes.
        
        Args:
            save_path (str): Path to save the PR curve plot
        """
        class_metrics = self.calculate_precision_recall_at_iou(0.5)
        
        plt.figure(figsize=(12, 8))
        
        for class_name, metrics in class_metrics.items():
            if metrics['total_gt'] > 0:
                plt.plot(metrics['recalls'], metrics['precisions'], 
                        label=f"{class_name} (GT: {metrics['total_gt']})", linewidth=2)
        
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves (IoU=0.5)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 1])
        plt.ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"PR curve saved to {save_path}")
    
    def evaluate(self, save_csv=False, output_dir='.'):
        """
        Perform complete evaluation and print results.
        
        Args:
            save_csv (bool): Whether to save per-class metrics as CSV
            output_dir (str): Directory to save output files
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("\n" + "="*60)
        print("OBJECT DETECTION EVALUATION RESULTS")
        print("="*60)
        
        map50_results = self.calculate_map([0.5])
        print(f"\nmAP50: {map50_results['mAP']:.4f}")
        
        iou_thresholds = np.arange(0.5, 0.96, 0.05)
        map50_95_results = self.calculate_map(iou_thresholds)
        print(f"mAP50-95: {map50_95_results['mAP']:.4f}")
        
        class_metrics = self.calculate_precision_recall_at_iou(0.5)
        
        print(f"\nPer-class results (IoU=0.5, Best F1 Score):")
        print("-" * 85)
        print(f"{'Class':<15} {'AP50':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Best conf':<10} {'GT':<6} {'Det':<6}")
        print("-" * 85)
        
        overall_precision_sum = 0
        overall_recall_sum = 0
        overall_f1_sum = 0
        valid_classes = 0
        overall_best_confs = []
        
        class_results = []
        
        for class_name in self.classes:
            metrics = class_metrics[class_name]
            ap50 = map50_results['class_mAPs'].get(class_name, 0.0)
            
            if metrics['total_gt'] > 0:
                best_precision = metrics['best_precision']
                best_recall = metrics['best_recall']
                best_f1 = metrics['best_f1']
                best_conf = metrics['best_conf']
                overall_precision_sum += best_precision
                overall_recall_sum += best_recall
                overall_f1_sum += best_f1
                valid_classes += 1
                overall_best_confs.append(best_conf)
            else:
                best_precision = 0.0
                best_recall = 0.0
                best_f1 = 0.0
                best_conf = 0.0
            
            print(f"{class_name:<15} {ap50:<8.4f} {best_precision:<10.4f} {best_recall:<8.4f} "
                  f"{best_f1:<8.4f} {best_conf:<10.4f} {metrics['total_gt']:<6} {metrics['total_detections']:<6}")
            
            class_results.append({
                'class': class_name,
                'AP50': ap50,
                'AP50-95': map50_95_results['class_mAPs'].get(class_name, 0.0),
                'precision': best_precision,
                'recall': best_recall,
                'f1': best_f1,
                'best_conf': best_conf,
                'total_gt': metrics['total_gt'],
                'total_detections': metrics['total_detections']
            })
        
        print("-" * 85)
        
        overall_precision = overall_precision_sum / valid_classes if valid_classes > 0 else 0.0
        overall_recall = overall_recall_sum / valid_classes if valid_classes > 0 else 0.0
        overall_f1 = overall_f1_sum / valid_classes if valid_classes > 0 else 0.0
        overall_best_conf = np.median(overall_best_confs) if valid_classes > 0 else 0.0
        
        print(f"\nOverall Metrics:")
        print(f"Overall Precision: {overall_precision:.4f}")
        print(f"Overall Recall: {overall_recall:.4f}")
        print(f"Overall F1: {overall_f1:.4f}")
        print(f"Overall best conf threshold: {overall_best_conf:.4f}")
        print(f"mAP50: {map50_results['mAP']:.4f}")
        print(f"mAP50-95: {map50_95_results['mAP']:.4f}")
        
        if save_csv:
            results_path = os.path.join(output_dir, 'evaluation_results.csv')
            results_df = pd.DataFrame(class_results)
            results_df.to_csv(results_path, index=False)
            print(f"\nPer-class metrics saved to {results_path}")
        
        pr_curve_path = os.path.join(output_dir, 'pr_curve.png')
        self.plot_pr_curve(pr_curve_path)
        
        return {
            'mAP50': map50_results['mAP'],
            'mAP50_95': map50_95_results['mAP'],
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_best_conf_threshold': overall_best_conf,
            'class_results': class_results
        }

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate object detection results')
    
    parser.add_argument('--gt', type=str, required=True,
                       help='Path to ground truth CSV file')
    parser.add_argument('--pred', type=str, required=True,
                       help='Path to predictions CSV file')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save output files (default: ./results)')
    parser.add_argument('--save_csv', type=bool, default=True,
                       help='Save per-class metrics to CSV file')
    parser.add_argument('--conf_thresholds', type=str, default='0.1,0.25,0.5,0.75,0.9',
                       help='Comma-separated confidence thresholds (default: 0.1,0.25,0.5,0.75,0.9)')
    
    return parser.parse_args()

def main():
    """Main function to run evaluation"""
    args = parse_arguments()
    
    print(f"Ground truth file: {args.gt}")
    print(f"Predictions file: {args.pred}")
    print(f"Output directory: {args.output_dir}")
    
    evaluator = ObjectDetectionEvaluator(args.gt, args.pred)
    results = evaluator.evaluate(save_csv=args.save_csv, output_dir=args.output_dir)
    
    print(f"\n" + "="*60)
    print("CONFIDENCE THRESHOLD ANALYSIS")
    print("="*60)
    
    try:
        conf_thresholds = [float(x.strip()) for x in args.conf_thresholds.split(',')]
    except ValueError:
        print("Warning: Invalid confidence thresholds format. Using defaults.")
        conf_thresholds = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    header = f"{'Conf':<6} {'Precision':<10} {'Recall':<8} {'F1':<8}"
    print(header)
    print("-" * len(header))
    
    for conf_thresh in conf_thresholds:
        class_metrics = evaluator.calculate_precision_recall_at_iou(0.5, conf_thresh)
        
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        valid_classes = 0
        
        for class_name, metrics in class_metrics.items():
            if metrics['total_gt'] > 0:
                total_precision += metrics['best_precision']
                total_recall += metrics['best_recall']
                total_f1 += metrics['best_f1']
                valid_classes += 1
        
        avg_precision = total_precision / valid_classes if valid_classes > 0 else 0.0
        avg_recall = total_recall / valid_classes if valid_classes > 0 else 0.0
        avg_f1 = total_f1 / valid_classes if valid_classes > 0 else 0.0
        
        print(f"{conf_thresh:<6} {avg_precision:<10.4f} {avg_recall:<8.4f} {avg_f1:<8.4f}")

if __name__ == "__main__":
    main()