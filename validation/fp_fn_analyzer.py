"""
Анализатор False Positives и False Negatives для детекции объектов
Сохраняет изображения с ошибками в отдельные папки для визуального анализа
"""

import pandas as pd
import cv2
import numpy as np
import os
from collections import defaultdict
import json
import argparse
from pathlib import Path


def calculate_iou(box1, box2):
    """Вычисление IoU между двумя bounding box"""
    x1 = max(box1['xmin'], box2['xmin'])
    y1 = max(box1['ymin'], box2['ymin'])
    x2 = min(box1['xmax'], box2['xmax'])
    y2 = min(box1['ymax'], box2['ymax'])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    area2 = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def draw_boxes_on_image(image, gt_boxes, pred_boxes, fps, fns):
    """Рисование боксов на изображении"""
    img = image.copy()
    
    # GT boxes - зеленые
    for box in gt_boxes:
        cv2.rectangle(img, 
                     (int(box['xmin']), int(box['ymin'])), 
                     (int(box['xmax']), int(box['ymax'])), 
                     (0, 255, 0), 2)
        cv2.putText(img, f"GT: {box['label']}", 
                   (int(box['xmin']), int(box['ymin'] - 5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # False Positives - красные
    for box in fps:
        cv2.rectangle(img, 
                     (int(box['xmin']), int(box['ymin'])), 
                     (int(box['xmax']), int(box['ymax'])), 
                     (0, 0, 255), 2)
        cv2.putText(img, f"FP: {box['detection_label']} ({box['conf']:.2f})", 
                   (int(box['xmin']), int(box['ymin'] - 5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # False Negatives - оранжевые пунктирные
    for box in fns:
        # Имитация пунктирной линии
        thickness = 2
        color = (0, 165, 255)  # оранжевый в BGR
        x1, y1, x2, y2 = int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])
        
        # Верхняя линия
        for i in range(x1, x2, 10):
            cv2.line(img, (i, y1), (min(i+5, x2), y1), color, thickness)
        # Нижняя линия
        for i in range(x1, x2, 10):
            cv2.line(img, (i, y2), (min(i+5, x2), y2), color, thickness)
        # Левая линия
        for i in range(y1, y2, 10):
            cv2.line(img, (x1, i), (x1, min(i+5, y2)), color, thickness)
        # Правая линия
        for i in range(y1, y2, 10):
            cv2.line(img, (x2, i), (x2, min(i+5, y2)), color, thickness)
            
        cv2.putText(img, f"FN: {box['label']}", 
                   (int(box['xmin']), int(box['ymax'] + 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img


def analyze_detections(gt_file, pred_file, output_dir, iou_threshold=0.5, conf_threshold=0.5, 
                      image_base_path="", class_filter=None, max_images_per_type=None):
    """Основная функция анализа"""
    
    print("Загрузка данных...")
    gt_df = pd.read_csv(gt_file)
    pred_df = pd.read_csv(pred_file)
    
    print(f"Загружено GT: {len(gt_df)} записей")
    print(f"Загружено predictions: {len(pred_df)} записей")
    
    # Фильтрация по confidence
    pred_df = pred_df[pred_df['conf'] >= conf_threshold]
    print(f"После фильтрации по confidence >= {conf_threshold}: {len(pred_df)} predictions")
    
    # Фильтрация по классам
    if class_filter:
        gt_df = gt_df[gt_df['label'] == class_filter]
        pred_df = pred_df[pred_df['detection_label'] == class_filter]
        print(f"Фильтрация по классу '{class_filter}': GT={len(gt_df)}, Pred={len(pred_df)}")
    
    # Группировка по изображениям
    gt_by_image = defaultdict(list)
    pred_by_image = defaultdict(list)
    
    for _, row in gt_df.iterrows():
        gt_by_image[row['image_path']].append(row.to_dict())
    
    for _, row in pred_df.iterrows():
        pred_by_image[row['image_path']].append(row.to_dict())
    
    # Создание выходных папок
    output_path = Path(output_dir)
    fp_dir = output_path / "false_positives"
    fn_dir = output_path / "false_negatives"
    both_dir = output_path / "both_errors"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)
    both_dir.mkdir(parents=True, exist_ok=True)
    
    # Анализ каждого изображения
    all_images = set(list(gt_by_image.keys()) + list(pred_by_image.keys()))
    
    results = {
        'fp_only': [],
        'fn_only': [],
        'both': [],
        'stats': {}
    }
    
    total_fp = 0
    total_fn = 0
    processed_images = 0
    
    print(f"\nОбработка {len(all_images)} изображений...")
    
    for image_path in all_images:
        gt_boxes = gt_by_image.get(image_path, [])
        pred_boxes = pred_by_image.get(image_path, [])
        
        # Поиск совпадений
        matched_pred = set()
        matched_gt = set()
        
        for pred_idx, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                
                if gt_box['label'] == pred_box['detection_label']:
                    iou = calculate_iou(gt_box, pred_box)
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_gt_idx = gt_idx
            
            if best_gt_idx != -1:
                matched_pred.add(pred_idx)
                matched_gt.add(best_gt_idx)
        
        # Определение FP и FN
        fps = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
        fns = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
        
        total_fp += len(fps)
        total_fn += len(fns)
        
        # Сохранение изображений с ошибками
        if fps or fns:
            processed_images += 1
            
            # Поиск изображения
            img_path = find_image_path(image_path, image_base_path)
            if img_path and os.path.exists(img_path):
                try:
                    image = cv2.imread(img_path)
                    if image is not None:
                        # Рисование боксов
                        annotated_img = draw_boxes_on_image(image, gt_boxes, pred_boxes, fps, fns)
                        
                        # Определение типа ошибки и сохранение
                        error_info = {
                            'image_path': image_path,
                            'fps': len(fps),
                            'fns': len(fns),
                            'fp_details': fps,
                            'fn_details': fns
                        }
                        
                        filename = os.path.basename(image_path)
                        base_name = os.path.splitext(filename)[0]
                        
                        if fps and fns:
                            # Оба типа ошибок
                            save_path = both_dir / f"{base_name}_FP{len(fps)}_FN{len(fns)}.jpg"
                            results['both'].append(error_info)
                        elif fps:
                            # Только FP
                            save_path = fp_dir / f"{base_name}_FP{len(fps)}.jpg"
                            results['fp_only'].append(error_info)
                        else:
                            # Только FN
                            save_path = fn_dir / f"{base_name}_FN{len(fns)}.jpg"
                            results['fn_only'].append(error_info)
                        
                        # Проверка лимита изображений
                        if max_images_per_type:
                            if (len(results['fp_only']) >= max_images_per_type and fps and not fns) or \
                               (len(results['fn_only']) >= max_images_per_type and fns and not fps) or \
                               (len(results['both']) >= max_images_per_type and fps and fns):
                                continue
                        
                        cv2.imwrite(str(save_path), annotated_img)
                        
                        # Сохранение JSON с деталями
                        json_path = save_path.with_suffix('.json')
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(error_info, f, indent=2, ensure_ascii=False)
                            
                except Exception as e:
                    print(f"Ошибка обработки {image_path}: {e}")
            else:
                print(f"Изображение не найдено: {image_path}")
    
    # Сохранение общей статистики
    results['stats'] = {
        'total_images': len(all_images),
        'processed_images': processed_images,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'fp_only_images': len(results['fp_only']),
        'fn_only_images': len(results['fn_only']),
        'both_errors_images': len(results['both']),
        'iou_threshold': iou_threshold,
        'conf_threshold': conf_threshold,
        'class_filter': class_filter
    }
    
    stats_path = output_path / "analysis_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(results['stats'], f, indent=2, ensure_ascii=False)
    
    # Вывод статистики
    print(f"\n=== РЕЗУЛЬТАТЫ АНАЛИЗА ===")
    print(f"Всего изображений: {len(all_images)}")
    print(f"Обработано изображений: {processed_images}")
    print(f"Всего False Positives: {total_fp}")
    print(f"Всего False Negatives: {total_fn}")
    print(f"Изображений только с FP: {len(results['fp_only'])}")
    print(f"Изображений только с FN: {len(results['fn_only'])}")
    print(f"Изображений с FP и FN: {len(results['both'])}")
    print(f"\nИзображения сохранены в:")
    print(f"  - False Positives: {fp_dir}")
    print(f"  - False Negatives: {fn_dir}")
    print(f"  - Оба типа ошибок: {both_dir}")
    print(f"  - Статистика: {stats_path}")


def find_image_path(image_path, base_path):
    """Поиск изображения в разных возможных местах"""
    possible_paths = [
        image_path,  # оригинальный путь
        os.path.join(base_path, image_path),  # с базовым путем
        os.path.join(base_path, os.path.basename(image_path)),  # только имя файла
        os.path.expanduser(image_path),  # раскрытие ~
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Анализ FP/FN детекции объектов")
    parser.add_argument("--gt", required=True, help="Путь к файлу gt.csv")
    parser.add_argument("--pred", required=True, help="Путь к файлу predictions.csv")
    parser.add_argument("--output", required=True, help="Папка для сохранения результатов")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold (default: 0.5)")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--image-path", default="", help="Базовый путь к изображениям")
    parser.add_argument("--class-filter", help="Фильтр по классу объектов")
    parser.add_argument("--max-images", type=int, help="Максимальное количество изображений каждого типа")
    
    args = parser.parse_args()
    
    analyze_detections(
        gt_file=args.gt,
        pred_file=args.pred,
        output_dir=args.output,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        image_base_path=args.image_path,
        class_filter=args.class_filter,
        max_images_per_type=args.max_images
    )


if __name__ == "__main__":
    main()