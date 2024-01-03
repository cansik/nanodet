import argparse
import os
import time
from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
import openvino as ov
from tqdm import tqdm


def current_millis() -> int:
    return time.time_ns() // 1_000_000


class Watch:
    def __init__(self, name: str = "Watch"):
        self.name = name
        self.start_time: int = 0
        self.end_time: int = 0
        self.running = False

    def start(self):
        self.start_time = current_millis()
        self.running = True

    def stop(self):
        self.end_time = current_millis()
        self.running = False

    def elapsed(self) -> int:
        if self.running:
            return current_millis() - self.start_time
        return self.end_time - self.start_time

    def time_str(self, time_format: str = "%Hh %Mm %Ss {}ms") -> str:
        delta = self.elapsed()
        return time.strftime(time_format.format(delta % 1000), time.gmtime(delta / 1000.0))

    def print(self):
        print(f"{self.name}: {self.time_str()}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.print()


def fast_exp(x):
    return np.exp(x)


def sigmoid(x):
    return 1.0 / (1.0 + fast_exp(-x))


def activation_function_softmax(src):
    alpha = np.max(src)
    dst = fast_exp(src - alpha)
    denominator = np.sum(dst)
    dst /= denominator
    return dst


def generate_grid_center_priors(input_height: int, input_width: int, strides: List[int]):
    center_priors = []
    for stride in strides:
        feat_w = np.ceil(input_width / stride).astype(int)
        feat_h = np.ceil(input_height / stride).astype(int)
        for y in range(feat_h):
            for x in range(feat_w):
                center_priors.append({"x": x, "y": y, "stride": stride})
    return center_priors


class NanoDet:
    def __init__(self, model_path: Union[str, os.PathLike]):
        model_path = Path(model_path)

        self.core = ov.Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model)
        self.input_shape = list(self.model.inputs[0].shape)

        self.num_class = 1
        self.reg_max = 7
        self.strides = [8, 16, 32, 64]
        self.input_size = (self.input_shape[3], self.input_shape[2])

        self.center_priors = generate_grid_center_priors(self.input_size[1], self.input_size[0], self.strides)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(image, 1.0, self.input_size, (0, 0, 0), swapRB=True, crop=False)

    def detect(self, image: np.ndarray, score_threshold: float, nms_threshold: float):
        with Watch("pre-processing"):
            input_data = self.preprocess(image)

        with Watch("inference"):
            infer_request = self.compiled_model.create_infer_request()
            input_tensor = ov.Tensor(array=input_data, shared_memory=True)
            infer_request.set_input_tensor(input_tensor)

            infer_request.start_async()
            infer_request.wait()

            output_tensor = infer_request.get_output_tensor()
            output = output_tensor.data

        with Watch("post-processing"):
            results = [[] for _ in range(self.num_class)]
            self.decode_infer(output, self.center_priors, score_threshold, results)

        dets = []
        for cls_results in results:
            self.nms(cls_results, nms_threshold)
            dets.extend(cls_results)

        # normalize output data
        w, h = self.input_size
        for det in dets:
            det["xmin"] /= w
            det["ymin"] /= h
            det["xmax"] /= w
            det["ymax"] /= h

        return dets

    def decode_infer(self, pred, center_priors, threshold, results):
        num_points, num_channels = pred.shape[1], pred.shape[2]
        for idx in range(num_points):
            ct_x, ct_y, stride = center_priors[idx]["x"], center_priors[idx]["y"], center_priors[idx]["stride"]
            score, cur_label = 0, 0
            for label in range(self.num_class):
                if pred[0, idx, label] > score:
                    score = pred[0, idx, label]
                    cur_label = label
            if score > threshold:
                bbox_pred = pred[0, idx, self.num_class:]
                results[cur_label].append(self.dis_pred_to_bbox(bbox_pred, cur_label, score, ct_x, ct_y, stride))

    def dis_pred_to_bbox(self, dfl_det, label, score, x, y, stride):
        ct_x, ct_y = x * stride, y * stride
        dis_pred = [0] * 4
        for i in range(4):
            dis = 0
            dis_after_sm = activation_function_softmax(dfl_det[i * (self.reg_max + 1):(i + 1) * (self.reg_max + 1)])
            for j in range(self.reg_max + 1):
                dis += j * dis_after_sm[j]
            dis *= stride
            dis_pred[i] = dis
        xmin = max(ct_x - dis_pred[0], 0)
        ymin = max(ct_y - dis_pred[1], 0)
        xmax = min(ct_x + dis_pred[2], self.input_size[1])
        ymax = min(ct_y + dis_pred[3], self.input_size[0])
        return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "score": score, "label": label}

    def nms(self, input_boxes, NMS_THRESH):
        input_boxes.sort(key=lambda x: x["score"], reverse=True)
        vArea = [(box["xmax"] - box["xmin"] + 1) * (box["ymax"] - box["ymin"] + 1) for box in input_boxes]
        i = 0
        while i < len(input_boxes):
            j = i + 1
            while j < len(input_boxes):
                xx1 = max(input_boxes[i]["xmin"], input_boxes[j]["xmin"])
                yy1 = max(input_boxes[i]["ymin"], input_boxes[j]["ymin"])
                xx2 = min(input_boxes[i]["xmax"], input_boxes[j]["xmax"])
                yy2 = min(input_boxes[i]["ymax"], input_boxes[j]["ymax"])
                w = max(0, xx2 - xx1 + 1)
                h = max(0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (vArea[i] + vArea[j] - inter)
                if ovr >= NMS_THRESH:
                    input_boxes.pop(j)
                    vArea.pop(j)
                else:
                    j += 1
            i += 1


def annotate_detections(image, detections, color=(0, 0, 255)):
    h, w = image.shape[:2]
    for det in detections:
        xmin, ymin, xmax, ymax, score, label = det["xmin"], det["ymin"], det["xmax"], det["ymax"], det["score"], det[
            "label"]
        cv2.rectangle(image, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to the openvino model.")
    parser.add_argument("input", type=str, help="Path to the image.")
    args = parser.parse_args()

    nanodet = NanoDet(args.model)

    input_path = Path(args.input)

    image_paths: List[Path] = []
    if input_path.is_file():
        image_paths.append(input_path)
    else:
        image_paths += list(input_path.glob("*.png"))
        image_paths += list(input_path.glob("*.jpeg"))
        image_paths += list(input_path.glob("*.jpg"))
        sorted(image_paths)

    for image_path in tqdm(image_paths, desc="Inference"):
        image = cv2.imread(str(image_path))

        with Watch(f"Total ({image_path.name})"):
            detections = nanodet.detect(image, 0.1, 0.1)

        annotate_detections(image, detections)

        cv2.imshow(f"Result", image)
        cv2.waitKey(0)

    exit(0)


if __name__ == "__main__":
    main()