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
    Оценка детекции прямой модели YOLO по списку картинок. Предсказывает, считает PR-кривые и метрики.
    """

    def __init__(self, weights, txt_path, output_dir, device=0, cfg=None):
        """
        Args:
            axis_weights (str): путь до весов осевой модели (axis).
            badaxis_weights (str): путь до весов модели badaxis.
            txt_path (str): путь до txt-файла со списком файлов-изображений (одно на строку).
            output_dir (str): путь к папке для сохранения результатов.
            device (int/str): cuda id или 'cpu'.
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
        Выполняет каскадное предсказание: сперва по axis, затем crop'ы -- по badaxis.

        Returns:
            list(dict): список предсказаний {'bbox':[x1, y1, x2, y2], 'conf':score, 'class':class_id}
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
        Основная функция: выполняет предсказания и сохраняет их.
        """
        # Читаем список файлов
        with open(self.txt_path) as f:
            img_paths = [line.strip() for line in f if line.strip()]

        predictions_df = defaultdict(list)

        # Предсказания и GT
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
    parser = argparse.ArgumentParser(description='Evaluator for straight YOLO models')
    parser.add_argument('--weights', type=str, required=True, help='Путь до весов модели (.pt)')
    parser.add_argument('--txt_path', type=str, required=True, help='Путь до файла с путями к картинкам (.txt)')
    parser.add_argument('--output_dir', type=str, default='results_eval', help='Куда сохранять метрики и графики')
    parser.add_argument('--device', type=str, default='0', help='CUDA id или "cpu"')
    parser.add_argument('--cfg', type=str, default=None, help='Путь до конфига модели')
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