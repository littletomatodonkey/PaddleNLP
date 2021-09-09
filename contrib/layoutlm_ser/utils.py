from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import shutil

import numpy as np


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
        "--infer_img",
        default=None,
        type=str,
        required=False, )

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
