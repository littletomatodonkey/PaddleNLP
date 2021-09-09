import os
import sys
sys.path.insert(0, "../../")

import json
import cv2
import numpy as np
from copy import deepcopy
import random

import paddle

# relative reference
from utils import parse_args, get_image_file_list, draw_ser_results
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification


# label to draw id
def get_label_to_id_map():
    label_to_id_map = {
        "O": 0,
        "B-QUESTION": 1,
        "I-QUESTION": 1,
        "B-ANSWER": 2,
        "I-ANSWER": 2,
        "B-HEADER": 3,
        "I-HEADER": 3,
    }
    return label_to_id_map


def get_labels():
    labels = [
        "O",
        "B-QUESTION",
        "B-ANSWER",
        "B-HEADER",
        "I-ANSWER",
        "I-QUESTION",
        "I-HEADER",
    ]
    return labels


# pad sentences
def pad_sentences(tokenizer,
                  encoded_inputs,
                  max_seq_len=512,
                  pad_to_max_seq_len=True,
                  return_attention_mask=True,
                  return_token_type_ids=True,
                  return_overflowing_tokens=False,
                  return_special_tokens_mask=False):
    # Padding with larger size, reshape is carried out
    max_seq_len = (
        len(encoded_inputs["input_ids"]) // max_seq_len + 1) * max_seq_len

    needs_to_be_padded = pad_to_max_seq_len and \
                         max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

    if needs_to_be_padded:
        difference = max_seq_len - len(encoded_inputs["input_ids"])
        if tokenizer.padding_side == 'right':
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"]) + [0] * difference
            if return_token_type_ids:
                encoded_inputs["token_type_ids"] = (
                    encoded_inputs["token_type_ids"] +
                    [tokenizer.pad_token_type_id] * difference)
            if return_special_tokens_mask:
                encoded_inputs["special_tokens_mask"] = encoded_inputs[
                    "special_tokens_mask"] + [1] * difference
            encoded_inputs["input_ids"] = encoded_inputs[
                "input_ids"] + [tokenizer.pad_token_id] * difference
            encoded_inputs["bbox"] = encoded_inputs["bbox"] + [[0, 0, 0, 0]
                                                               ] * difference
    else:
        if return_attention_mask:
            encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                "input_ids"])

    return encoded_inputs


def truncate_inputs(encoded_inputs, max_seq_len=512):
    """
    truncate is often used in training process
    """
    for key in encoded_inputs:
        length = min(len(encoded_inputs[key]), max_seq_len)
        encoded_inputs[key] = encoded_inputs[key][:length]
    return encoded_inputs


def split_page(encoded_inputs, max_seq_len=512):
    """
    truncate is often used in training process
    """
    for key in encoded_inputs:
        encoded_inputs[key] = paddle.to_tensor(encoded_inputs[key])
        if encoded_inputs[key].ndim <= 1:  # for input_ids, att_mask and so on
            encoded_inputs[key] = encoded_inputs[key].reshape([-1, max_seq_len])
        else:  # for bbox
            encoded_inputs[key] = encoded_inputs[key].reshape(
                [-1, max_seq_len, 4])
    return encoded_inputs


def preprocess(
        tokenizer,
        ori_img,
        ocr_info,
        img_size=(224, 224),
        pad_token_label_id=-100,
        max_seq_len=512,
        add_special_ids=False,
        return_attention_mask=True, ):
    ocr_info = deepcopy(ocr_info)
    height = ori_img.shape[0]
    width = ori_img.shape[1]

    img = cv2.resize(ori_img,
                     (224, 224)).transpose([2, 0, 1]).astype(np.float32)

    segment_offset_id = []
    words_list = []
    bbox_list = []
    input_ids_list = []
    token_type_ids_list = []

    for info in ocr_info:
        # x1, y1, x2, y2
        bbox = info["bbox"]
        bbox[0] = int(bbox[0] * 1000.0 / width)
        bbox[2] = int(bbox[2] * 1000.0 / width)
        bbox[1] = int(bbox[1] * 1000.0 / height)
        bbox[3] = int(bbox[3] * 1000.0 / height)

        text = info["text"]
        encode_res = tokenizer.encode(
            text, pad_to_max_seq_len=False, return_attention_mask=True)

        if not add_special_ids:
            # TODO: use tok.all_special_ids to remove
            encode_res["input_ids"] = encode_res["input_ids"][1:-1]
            encode_res["token_type_ids"] = encode_res["token_type_ids"][1:-1]
            encode_res["attention_mask"] = encode_res["attention_mask"][1:-1]

        input_ids_list.extend(encode_res["input_ids"])
        token_type_ids_list.extend(encode_res["token_type_ids"])
        bbox_list.extend([bbox] * len(encode_res["input_ids"]))
        words_list.append(text)
        segment_offset_id.append(len(input_ids_list))

    encoded_inputs = {
        "input_ids": input_ids_list,
        "token_type_ids": token_type_ids_list,
        "bbox": bbox_list,
        "attention_mask": [1] * len(input_ids_list),
    }

    encoded_inputs = pad_sentences(
        tokenizer,
        encoded_inputs,
        max_seq_len=max_seq_len,
        return_attention_mask=return_attention_mask)

    # encoded_inputs = truncate_inputs(encoded_inputs)

    encoded_inputs = split_page(encoded_inputs)

    fake_bs = encoded_inputs["input_ids"].shape[0]

    encoded_inputs["image"] = paddle.to_tensor(img).unsqueeze(0).expand(
        [fake_bs] + list(img.shape))

    encoded_inputs["segment_offset_id"] = segment_offset_id

    return encoded_inputs


def postprocess(attention_mask, preds):
    if isinstance(preds, paddle.Tensor):
        preds = preds.numpy()
    preds = np.argmax(preds, axis=2)

    labels = get_labels()

    label_map = {i: label.upper() for i, label in enumerate(labels)}

    preds_list = [[] for _ in range(preds.shape[0])]

    # keep batch info for the 
    for i in range(preds.shape[0]):
        for j in range(preds.shape[1]):
            if attention_mask[i][j] == 1:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def merge_preds_list_with_ocr_info(ocr_info, segment_offset_id, preds_list):
    # must ensure the preds_list is generated from the same image
    preds = [p for pred in preds_list for p in pred]
    label_to_id_map = get_label_to_id_map()

    for idx in range(len(segment_offset_id)):
        if idx == 0:
            start_id = 0
        else:
            start_id = segment_offset_id[idx - 1]

        end_id = segment_offset_id[idx]

        curr_pred = preds[start_id:end_id]
        curr_pred = [label_to_id_map[p] for p in curr_pred]

        if len(curr_pred) <= 0:
            pred_id = 0
        else:
            counts = np.bincount(curr_pred)
            pred_id = np.argmax(counts)
        ocr_info[idx]["pred_id"] = pred_id
    return ocr_info


@paddle.no_grad()
def infer(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # init token and model
    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)
    # model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
    model = LayoutXLMForTokenClassification.from_pretrained(
        args.model_name_or_path)
    model.eval()

    # load ocr results json
    ocr_results = dict()
    with open(args.ocr_json_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            img_name, json_info = line.split("\t")
            ocr_results[os.path.basename(img_name)] = json.loads(json_info)

    # get infer img list
    infer_imgs = get_image_file_list(args.infer_imgs)

    # loop for infer
    for idx, img_path in enumerate(infer_imgs):
        print("process: [{}/{}]".format(idx, len(infer_imgs), img_path))

        img = cv2.imread(img_path)

        ocr_info = ocr_results[os.path.basename(img_path)]["ocr_info"]
        inputs = preprocess(
            tokenizer=tokenizer,
            ori_img=img,
            ocr_info=ocr_info,
            max_seq_len=args.max_seq_length)

        outputs = model(
            input_ids=inputs["input_ids"],
            bbox=inputs["bbox"],
            image=inputs["image"],
            token_type_ids=inputs["token_type_ids"],
            attention_mask=inputs["attention_mask"])

        preds = outputs[0]
        preds = postprocess(inputs["attention_mask"], preds)
        ocr_info = merge_preds_list_with_ocr_info(
            ocr_info, inputs["segment_offset_id"], preds)

        img_res = draw_ser_results(img, ocr_info)
        cv2.imwrite(
            os.path.join(args.output_dir, os.path.basename(img_path)), img_res)

    return


if __name__ == "__main__":
    args = parse_args()
    infer(args)
