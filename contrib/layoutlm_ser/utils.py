from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil
import imghdr

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def get_label_maps(label_map_path=None):
    with open(label_map_path, "r") as fin:
        lines = fin.readlines()
    labels = [line.strip() for line in lines]
    label2id_map = {label: idx for idx, label in enumerate(labels)}
    id2label_map = {idx: label for idx, label in enumerate(labels)}
    return label2id_map, id2label_map


def get_image_file_list(img_file):
    imgs_lists = []
    if img_file is None or not os.path.exists(img_file):
        raise Exception("not found any img file in {}".format(img_file))

    img_end = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff', 'gif', 'GIF'}
    if os.path.isfile(img_file) and imghdr.what(img_file) in img_end:
        imgs_lists.append(img_file)
    elif os.path.isdir(img_file):
        for single_file in os.listdir(img_file):
            file_path = os.path.join(img_file, single_file)
            if os.path.isfile(file_path) and imghdr.what(file_path) in img_end:
                imgs_lists.append(file_path)
    if len(imgs_lists) == 0:
        raise Exception("not found any img file in {}".format(img_file))
    imgs_lists = sorted(imgs_lists)
    return imgs_lists


def draw_ser_results(image, ocr_results):

    color_map = {
        1: (0, 0, 255),  # question
        2: (0, 255, 0),  # answer
        3: (255, 0, 0),  # header
    }
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    img_new = image.copy()
    draw = ImageDraw.Draw(img_new)
    for ocr_info in ocr_results:
        if ocr_info["pred_id"] not in color_map:
            continue
        color = color_map[ocr_info["pred_id"]]

        # just for rectangle
        bbox = ocr_info["bbox"]
        bbox = ((bbox[0], bbox[1]), (bbox[2], bbox[3]))
        draw.rectangle(bbox, fill=color)

    img_new = Image.blend(image, img_new, 0.5)
    return np.array(img_new)


def parse_args():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True, )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True, )
    parser.add_argument(
        "--ocr_rec_model_dir",
        default="./ch_ppocr_mobile_v2.1_rec_infer",
        type=str, )
    parser.add_argument(
        "--ocr_det_model_dir",
        default="./ch_ppocr_mobile_v2.1_det_infer",
        type=str, )
    parser.add_argument(
        "--weights_path",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--train_data_dir",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--train_label_path",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--eval_data_dir",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--eval_label_path",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--label_map_path",
        default="./labels/labels_ser.txt",
        type=str,
        required=False, )

    parser.add_argument(
        "--infer_imgs",
        default=None,
        type=str,
        required=False, )

    parser.add_argument(
        "--ocr_json_path",
        default=None,
        type=str,
        required=False,
        help="ocr prediction results")

    parser.add_argument(
        "--use_vdl",
        default=False,
        type=bool,
        required=False, )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name", )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.", )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.", )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.", )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.")
    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.")
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3,
        type=int,
        help="Total number of training epochs to perform.", )
    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.")

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=10,
        help="eval every X updates steps.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.", )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    return args
