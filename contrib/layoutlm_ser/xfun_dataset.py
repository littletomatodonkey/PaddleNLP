import json
import random
import os
import sys
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader

__all__ = ["XfunDatasetForSer"]


class XfunDatasetForSer(Dataset):
    """
    Example:
        print("=====begin to build dataset=====")
        from paddlenlp.transformers import LayoutXLMTokenizer
        tokenizer = LayoutXLMTokenizer.from_pretrained("/paddle/models/transformers/layoutxlm-base-paddle/")
        tok_res = tokenizer.tokenize("Maribyrnong")
        # res = tokenizer.convert_ids_to_tokens(val_data["input_ids"][0])
        dataset = XfunDatasetForSer(
            tokenizer,
            data_dir="./zh.val/",
            label_path="zh.val/xfun_normalize_val.json",
            img_size=(224,224))
        print(len(dataset))

        data = dataset[0]
        print(data.keys())
        print("input_ids: ", data["input_ids"])
        print("label_ids: ", data["label_ids"])
        print("token_type_ids: ", data["token_type_ids"])
        print("words_list: ", data["words_list"])
        print("image shape: ", data["image"].shape)
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            label_path,
            img_size=(224, 224),
            pad_token_label_id=None,
            add_special_ids=False,
            return_attention_mask=True, ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.add_special_ids = add_special_ids
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.all_lines = self.read_all_lines()

        self.label_to_id = {
            "other": 0,
            "b-question": 1,
            "b-answer": 2,
            "b-header": 3,
            "i-answer": 4,
            "i-question": 5,
            "i-header": 6,
        }

    def pad_sentences(self,
                      encoded_inputs,
                      max_seq_len=512,
                      pad_to_max_seq_len=True,
                      return_attention_mask=True,
                      return_token_type_ids=True,
                      truncation_strategy="longest_first",
                      return_overflowing_tokens=False,
                      return_special_tokens_mask=False):
        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.tokenizer.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.tokenizer.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["label_ids"] = encoded_inputs[
                    "label_ids"] + [self.pad_token_label_id] * difference
                encoded_inputs["bbox"] = encoded_inputs[
                    "bbox"] + [[0, 0, 0, 0]] * difference
            elif self.tokenizer.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.tokenizer.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
                encoded_inputs["label_ids"] = [
                    self.pad_token_label_id
                ] * difference + encoded_inputs["label_ids"]
                encoded_inputs["bbox"] = [
                    [0, 0, 0, 0]
                ] * difference + encoded_inputs["bbox"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        return encoded_inputs

    def truncate_inputs(self, encoded_inputs, max_seq_len=512):
        for key in encoded_inputs:
            length = min(len(encoded_inputs[key]), max_seq_len)
            encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r") as fin:
            lines = fin.readlines()
        return lines

    def parse_label_file(self, line):
        image_name, info_str = line.split("\t")
        image_path = os.path.join(self.data_dir, image_name)

        # read img
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])

        # read text info
        info_dict = json.loads(info_str)
        height = info_dict["height"]
        width = info_dict["width"]

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        gt_label_list = []

        for info in info_dict["ocr_info"]:
            # x1, y1, x2, y2
            bbox = info["bbox"]
            label = info["label"]
            bbox[0] = int(bbox[0] * 1000.0 / width)
            bbox[2] = int(bbox[2] * 1000.0 / width)
            bbox[1] = int(bbox[1] * 1000.0 / height)
            bbox[3] = int(bbox[3] * 1000.0 / height)

            text = info["text"]
            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)

            gt_label = []
            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:
                                                                            -1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:
                                                                            -1]
            if label.lower() == "other":
                gt_label.extend([0] * len(encode_res["input_ids"]))
            else:
                gt_label.append(self.label_to_id[("b-" + label).lower()])
                gt_label.extend([self.label_to_id[("i-" + label).lower()]] *
                                (len(encode_res["input_ids"]) - 1))

            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            gt_label_list.extend(gt_label)
            words_list.append(text)

        encoded_inputs = {
            "input_ids": input_ids_list,
            "label_ids": gt_label_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
            # "words_list": words_list,
        }

        encoded_inputs = self.pad_sentences(
            encoded_inputs, return_attention_mask=self.return_attention_mask)
        encoded_inputs = self.truncate_inputs(encoded_inputs)

        encoded_inputs["image"] = img

        res = [
            np.array(
                encoded_inputs["input_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["label_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["token_type_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["bbox"], dtype=np.int64),
            np.array(
                encoded_inputs["attention_mask"], dtype=np.int64),
            np.array(
                encoded_inputs["image"], dtype=np.float32),
        ]
        return res

    def __getitem__(self, idx):
        res = self.parse_label_file(self.all_lines[idx])
        return res
        try:
            res = self.parse_label_file(self.all_lines[idx])
            return res
        except Exception as ex:
            print(ex)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self, ):
        return len(self.all_lines)


class XfunDatasetInfer(XfunDatasetForSer):
    """
    Example:
        print("=====begin to build dataset=====")
        from paddlenlp.transformers import LayoutXLMTokenizer
        tokenizer = LayoutXLMTokenizer.from_pretrained("/paddle/models/transformers/layoutxlm-base-paddle/")
        tok_res = tokenizer.tokenize("Maribyrnong")
        # res = tokenizer.convert_ids_to_tokens(val_data["input_ids"][0])
        dataset = XfunDatasetForSer(
            tokenizer,
            data_dir="./zh.val/",
            label_path="zh.val/xfun_normalize_val.json",
            img_size=(224,224))
        print(len(dataset))

        data = dataset[0]
        print(data.keys())
        print("input_ids: ", data["input_ids"])
        print("label_ids: ", data["label_ids"])
        print("token_type_ids: ", data["token_type_ids"])
        print("words_list: ", data["words_list"])
        print("image shape: ", data["image"].shape)
    """

    def __init__(
            self,
            tokenizer,
            data_dir,
            label_path,
            img_size=(224, 224),
            pad_token_label_id=None,
            add_special_ids=False,
            return_attention_mask=True, ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.add_special_ids = add_special_ids
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.all_lines = self.read_all_lines()

        self.label_to_id = {
            "other": 0,
            "b-question": 1,
            "b-answer": 2,
            "b-header": 3,
            "i-answer": 4,
            "i-question": 5,
            "i-header": 6,
        }

    def pad_sentences(self,
                      encoded_inputs,
                      max_seq_len=512,
                      pad_to_max_seq_len=True,
                      return_attention_mask=True,
                      return_token_type_ids=True,
                      truncation_strategy="longest_first",
                      return_overflowing_tokens=False,
                      return_special_tokens_mask=False):
        # Padding
        needs_to_be_padded = pad_to_max_seq_len and \
                             max_seq_len and len(encoded_inputs["input_ids"]) < max_seq_len

        if needs_to_be_padded:
            difference = max_seq_len - len(encoded_inputs["input_ids"])
            if self.tokenizer.padding_side == 'right':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                        "input_ids"]) + [0] * difference
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] +
                        [self.tokenizer.pad_token_type_id] * difference)
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs[
                        "special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs[
                    "input_ids"] + [self.tokenizer.pad_token_id] * difference
                encoded_inputs["label_ids"] = encoded_inputs[
                    "label_ids"] + [self.pad_token_label_id] * difference
                encoded_inputs["bbox"] = encoded_inputs[
                    "bbox"] + [[0, 0, 0, 0]] * difference
            elif self.tokenizer.padding_side == 'left':
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [
                        1
                    ] * len(encoded_inputs["input_ids"])
                if return_token_type_ids:
                    encoded_inputs["token_type_ids"] = (
                        [self.tokenizer.pad_token_type_id] * difference +
                        encoded_inputs["token_type_ids"])
                if return_special_tokens_mask:
                    encoded_inputs["special_tokens_mask"] = [
                        1
                    ] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * difference + encoded_inputs["input_ids"]
                encoded_inputs["label_ids"] = [
                    self.pad_token_label_id
                ] * difference + encoded_inputs["label_ids"]
                encoded_inputs["bbox"] = [
                    [0, 0, 0, 0]
                ] * difference + encoded_inputs["bbox"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        return encoded_inputs

    def truncate_inputs(self, encoded_inputs, max_seq_len=512):
        for key in encoded_inputs:
            length = min(len(encoded_inputs[key]), max_seq_len)
            encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r") as fin:
            lines = fin.readlines()
        return lines

    def parse_label_file(self, line):
        image_name, info_str = line.split("\t")
        image_path = os.path.join(self.data_dir, image_name)

        # read img
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])

        # read text info
        info_dict = json.loads(info_str)
        height = info_dict["height"]
        width = info_dict["height"]

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        gt_label_list = []

        for info in info_dict["ocr_info"]:
            # x1, y1, x2, y2
            bbox = info["bbox"]
            label = info["label"]
            bbox[0] = int(bbox[0] * 1000.0 / width)
            bbox[2] = int(bbox[2] * 1000.0 / width)
            bbox[1] = int(bbox[1] * 1000.0 / height)
            bbox[3] = int(bbox[3] * 1000.0 / height)

            text = info["text"]
            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)

            gt_label = []
            if not self.add_special_ids:
                # TODO: use tok.all_special_ids to remove
                encode_res["input_ids"] = encode_res["input_ids"][1:-1]
                encode_res["token_type_ids"] = encode_res["token_type_ids"][1:
                                                                            -1]
                encode_res["attention_mask"] = encode_res["attention_mask"][1:
                                                                            -1]
            if label.lower() == "other":
                gt_label.extend([0] * len(encode_res["input_ids"]))
            else:
                gt_label.append(self.label_to_id[("b-" + label).lower()])
                gt_label.extend([self.label_to_id[("i-" + label).lower()]] *
                                (len(encode_res["input_ids"]) - 1))

            input_ids_list.extend(encode_res["input_ids"])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            gt_label_list.extend(gt_label)
            words_list.append(text)

        encoded_inputs = {
            "input_ids": input_ids_list,
            "label_ids": gt_label_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
            # "words_list": words_list,
        }

        encoded_inputs = self.pad_sentences(
            encoded_inputs, return_attention_mask=self.return_attention_mask)
        encoded_inputs = self.truncate_inputs(encoded_inputs)

        encoded_inputs["image"] = img

        res = [
            np.array(
                encoded_inputs["input_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["label_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["token_type_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["bbox"], dtype=np.int64),
            np.array(
                encoded_inputs["attention_mask"], dtype=np.int64),
            np.array(
                encoded_inputs["image"], dtype=np.float32),
        ]
        return res

    def __getitem__(self, idx):
        res = self.parse_label_file(self.all_lines[idx])
        return res
        try:
            res = self.parse_label_file(self.all_lines[idx])
            return res
        except Exception as ex:
            print(ex)
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self, ):
        return len(self.all_lines)
