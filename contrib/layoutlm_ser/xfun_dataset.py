import json
import random
import os
import sys
import cv2
import numpy as np
import paddle
import copy
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
            label2id_map=None,
            img_size=(224, 224),
            pad_token_label_id=None,
            add_special_ids=False,
            return_attention_mask=True, 
            model_type='layoutxlm',
            load_mode='all'):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.add_special_ids = add_special_ids
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.all_lines = self.read_all_lines()

        self.label2id_map = label2id_map
        self.model_type = model_type
        self.load_mode = load_mode
        if load_mode == "all":
            self.encoded_inputs_all = self.read_encoded_inputs_all()        

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
            if key == "sample_id":
                continue
            length = min(len(encoded_inputs[key]), max_seq_len)
            encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r") as fin:
            lines = fin.readlines()
        return lines
    
    def read_encoded_inputs_sample(self, info_str):
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
                gt_label.append(self.label2id_map[("b-" + label).upper()])
                gt_label.extend([self.label2id_map[("i-" + label).upper()]] *
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
        return encoded_inputs

    def read_encoded_inputs_all(self):
        num_samples = len(self.all_lines)
        encoded_inputs_all = []
        for lno in range(num_samples):
            line = self.all_lines[lno]
            image_name, info_str = line.split("\t")
            encoded_inputs = self.read_encoded_inputs_sample(info_str)
            seq_len = len(encoded_inputs['input_ids'])
            chunk_size = 512
            for chunk_id, index in enumerate(range(0, seq_len, chunk_size)):
                chunk_beg = index
                chunk_end = min(index + chunk_size, seq_len)
                encoded_inputs_example = {}
                for key in encoded_inputs:
                    encoded_inputs_example[key] = encoded_inputs[key][chunk_beg:chunk_end]
                encoded_inputs_example['sample_id'] = lno
                encoded_inputs_all.append(encoded_inputs_example)
        return encoded_inputs_all
    
    def parse_label_file(self, line):
        image_name, info_str = line.split("\t")
        image_path = os.path.join(self.data_dir, image_name)

        # read img
        if self.model_type == "layoutxlm-pp":
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resize_h, resize_w = 224, 224
            im_shape = img.shape[0:2]
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]
            img_new = cv2.resize(img, None, None, 
                                 fx=im_scale_x, fy=im_scale_y, 
                                 interpolation=2)
            mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
            std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
            img_new = img_new / 255.0
            img_new -= mean
            img_new /= std
            img = img_new.transpose((2, 0, 1))
        else:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])

        encoded_inputs = self.read_encoded_inputs_sample(info_str)
        
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

    def parse_label_file_all(self, idx):
        sample_id = self.encoded_inputs_all[idx]['sample_id']
        image_name, info_str = self.all_lines[sample_id].split("\t")
        image_path = os.path.join(self.data_dir, image_name)

        # read img
        if self.model_type == "layoutxlm-pp":
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resize_h, resize_w = 224, 224
            im_shape = img.shape[0:2]
            im_scale_y = resize_h / im_shape[0]
            im_scale_x = resize_w / im_shape[1]
            img_new = cv2.resize(img, None, None, 
                                 fx=im_scale_x, fy=im_scale_y, 
                                 interpolation=2)
            mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
            std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
            img_new = img_new / 255.0
            img_new -= mean
            img_new /= std
            img = img_new.transpose((2, 0, 1))
        else:
            img = cv2.imread(image_path)
            img = cv2.resize(img, (224, 224)).transpose([2, 0, 1])

        encoded_inputs = copy.deepcopy(self.encoded_inputs_all[idx])
        
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
        if self.load_mode == "all":
            res = self.parse_label_file_all(idx)
            return res            
        else:
            res = self.parse_label_file(self.all_lines[idx])
            return res

    def __len__(self, ):
        if self.load_mode == "all":
            return len(self.encoded_inputs_all)
        else:
            return len(self.all_lines)
