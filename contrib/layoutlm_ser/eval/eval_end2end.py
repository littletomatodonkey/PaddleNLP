import os
import sys

import numpy as np
import copy

import json


def get_labels(label_path):
    with open(label_path, "r") as fin:
        lines = fin.readlines()
    ori_labels = [line.strip() for line in lines]
    labels = ["O"]
    for key in ori_labels:
        if key.startswith("B-"):
            labels.append(key[2:])
    return labels


def parse_ser_results_fp(fp, is_gt=False):
    # img/zh_val_0.jpg        {
    #     "height": 3508,
    #     "width": 2480,
    #     "ocr_info": [
    #         {"text": "Maribyrnong", "label": "other", "bbox": [1958, 144, 2184, 198]},
    #         {"text": "CITYCOUNCIL", "label": "other", "bbox": [2052, 183, 2171, 214]},
    #     ]
    res_dict = dict()
    with open(fp, "r") as fin:
        lines = fin.readlines()

    for idx, line in enumerate(lines):
        label_text_maps = dict()
        img_path, info = line.strip().split("\t")
        image_name = os.path.basename(img_path)
        json_info = json.loads(info)
        for single_ocr_info in json_info["ocr_info"]:
            if is_gt:
                label = single_ocr_info["label"].upper()
            else:
                label = single_ocr_info["pred"].upper()
            if label in ["O", "OTHERS", "OTHER"]:
                label = "O"
            text = single_ocr_info["text"]

            if label not in label_text_maps:
                label_text_maps[label] = set()
            label_text_maps[label].add(text)
        res_dict[image_name] = copy.deepcopy(label_text_maps)

        break

    return res_dict


def calc_metrics(pred_res_map, gt_res_map, labels):
    total_tp_cnt = {label: 0 for label in labels if label != "O"}
    total_fp_cnt = {label: 0 for label in labels if label != "O"}
    total_fn_cnt = {label: 0 for label in labels if label != "O"}

    pred_fn_set = set(pred_res_map.keys())
    gt_fn_set = set(gt_res_map.keys())

    assert pred_fn_set == gt_fn_set

    for image_name in pred_res_map:
        tp, fp, fn = 0, 0, 0
        pred_ocr_info = pred_res_map[image_name]
        gt_ocr_info = gt_res_map[image_name]

        for label in pred_ocr_info:
            if label not in total_tp_cnt:
                continue
            pred_set_for_label = pred_ocr_info[label]
            gt_set_for_label = gt_ocr_info[label]
            tp = len(pred_set_for_label & gt_set_for_label)
            fp = len(pred_set_for_label - gt_set_for_label)
            fn = len(gt_set_for_label - pred_set_for_label)
            total_tp_cnt[label] += tp
            total_fp_cnt[label] += fp
            total_fn_cnt[label] += fn

    avg_precision = {label: 0 for label in labels if label != "O"}
    avg_recall = {label: 0 for label in labels if label != "O"}
    avg_f1 = {label: 0 for label in labels if label != "O"}

    for label in avg_precision:
        avg_precision[label] = 1.0 * total_tp_cnt[label] / (
            total_tp_cnt[label] + total_fp_cnt[label])
        avg_recall[label] = 1.0 * total_tp_cnt[label] / (
            total_tp_cnt[label] + total_fn_cnt[label])
        avg_f1[label] = 2 / (
            1.0 / avg_precision[label] + 1.0 / avg_recall[label])

    avg_precision["avg"] = float(np.mean(list(avg_precision.values())))
    avg_recall["avg"] = float(np.mean(list(avg_recall.values())))
    avg_f1["avg"] = float(np.mean(list(avg_f1.values())))

    metrics = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
    }
    return metrics


def eval_ser(pred_file, gt_file, label_path):
    pred_res_map = parse_ser_results_fp(pred_file, is_gt=False)
    gt_res_map = parse_ser_results_fp(gt_file, is_gt=True)

    labels = get_labels(label_path)

    metrics = calc_metrics(pred_res_map, gt_res_map, labels)
    print(metrics)

    return metrics


if __name__ == "__main__":
    pred_file = sys.argv[1]
    gt_file = sys.argv[2]
    label_path = sys.argv[3]
    eval_ser(pred_file, gt_file, label_path)
