import json
import random
import os
import sys
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
import random

__all__ = ["DatasetForPretrain"]

import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

class DatasetForPretrain(Dataset):
    """
    Example:
        print("=====begin to build dataset=====")
        from paddlenlp.transformers import LayoutXLMTokenizer
        tokenizer = LayoutXLMTokenizer.from_pretrained("/paddle/models/transformers/layoutxlm-base-paddle/")
        tok_res = tokenizer.tokenize("Maribyrnong")
        # res = tokenizer.convert_ids_to_tokens(val_data["input_ids"][0])
        dataset = XfunDatasetxForSer(
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
            return_attention_mask=True,
            model_type='layoutxlm'):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.add_special_ids = add_special_ids
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.all_lines = self.read_all_lines()
        self.model_type = model_type

#         self.label_to_id = {
#             "other": 0,
#             "b-question": 1,
#             "b-answer": 2,
#             "b-header": 3,
#             "i-answer": 4,
#             "i-question": 5,
#             "i-header": 6,
#         }

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
#                 encoded_inputs["label_ids"] = encoded_inputs[
#                     "label_ids"] + [self.pad_token_label_id] * difference
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
#                 encoded_inputs["label_ids"] = [
#                     self.pad_token_label_id
#                 ] * difference + encoded_inputs["label_ids"]
                encoded_inputs["bbox"] = [
                    [0, 0, 0, 0]
                ] * difference + encoded_inputs["bbox"]
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs[
                    "input_ids"])

        return encoded_inputs

    def truncate_inputs(self, encoded_inputs, max_seq_len=512):
        seq_len = len(encoded_inputs["input_ids"])
        if seq_len > max_seq_len:
            seq_beg_no = random.randint(0, seq_len-max_seq_len-1)
            seq_end_no = seq_beg_no + max_seq_len
        else:
            seq_beg_no = 0
            seq_end_no = seq_len
            
        for key in encoded_inputs:
            encoded_inputs[key] = encoded_inputs[key][seq_beg_no:seq_end_no]
            
#         for key in encoded_inputs:
#             length = min(len(encoded_inputs[key]), max_seq_len)
#             encoded_inputs[key] = encoded_inputs[key][:length]
        return encoded_inputs

    def read_all_lines(self, ):
        with open(self.label_path, "r") as fin:
            lines = fin.readlines()
        return lines

    def parse_label_file(self, line):
        info_dict = json.loads(line)
        image_name = info_dict['img_path']
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

        # read text info
#         info_dict = json.loads(info_str)
#         height = info_dict["height"]
#         width = info_dict["width"]
        height = info_dict['shape'][0]
        width = info_dict['shape'][1]

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        masked_lm_positions_list = []
        masked_lm_ids_list = []
#         gt_label_list = []

        for info in info_dict["info"]:
            # x1, y1, x2, y2
            bbox = np.array(info["box"])
            text = info["text"]
            left = int(min(bbox[:, 0]))
            right = int(max(bbox[:, 0]))
            top = int(min(bbox[:, 1]))
            bottom = int(max(bbox[:, 1]))
            bbox = [left, top, right, bottom]
            bbox[0] = int(bbox[0] * 1000.0 / width)
            bbox[2] = int(bbox[2] * 1000.0 / width)
            bbox[1] = int(bbox[1] * 1000.0 / height)
            bbox[3] = int(bbox[3] * 1000.0 / height)
            encode_res = self.tokenizer.encode(
                text, pad_to_max_seq_len=False, return_attention_mask=True)

#             gt_label = []
#             if not self.add_special_ids:
#                 # TODO: use tok.all_special_ids to remove
#                 encode_res["input_ids"] = encode_res["input_ids"][1:-1]
#                 encode_res["token_type_ids"] = encode_res["token_type_ids"][1:
#                                                                             -1]
#                 encode_res["attention_mask"] = encode_res["attention_mask"][1:
#                                                                             -1]
#             if label.lower() == "other":
#                 gt_label.extend([0] * len(encode_res["input_ids"]))
#             else:
#                 gt_label.append(self.label_to_id[("b-" + label).lower()])
#                 gt_label.extend([self.label_to_id[("i-" + label).lower()]] *
#                                 (len(encode_res["input_ids"]) - 1))

            input_ids_list.extend(encode_res['input_ids'])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))
            
        encoded_inputs = {
            "input_ids": input_ids_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
        }
        
        encoded_inputs = self.pad_sentences(
            encoded_inputs, return_attention_mask=self.return_attention_mask)
        encoded_inputs = self.truncate_inputs(encoded_inputs)

        #####create_masked_lm_predictions
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        mask_token_id = self.tokenizer.mask_token_id
        pad_token_id = self.tokenizer.pad_token_id
        cand_indexes = []
        for (i, ids) in enumerate(encoded_inputs["input_ids"]):
            if ids in [cls_token_id, sep_token_id, mask_token_id, pad_token_id]:
                continue
            cand_indexes.append(i)
            
        random.shuffle(cand_indexes)
        output_ids = list(encoded_inputs["input_ids"])
        masked_lm_prob = 0.15
        max_predictions_per_seq = 20
        num_to_predict = min(max_predictions_per_seq,
            max(1, int(round(len(cand_indexes) * masked_lm_prob))))
        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_id = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_id = mask_token_id
            else:
                # 20% of the time, keep original
                masked_id = encoded_inputs["input_ids"][index]
            output_ids[index] = masked_id

            masked_lms.append(MaskedLmInstance(index=index,
                label=encoded_inputs["input_ids"][index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_ids = []
        for p in masked_lms:
#             print(p.index, p.label)
            masked_lm_positions.append(p.index)
            masked_lm_ids.append(p.label)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)

        #####end create_masked_lm_predictions
#         print(encoded_inputs["input_ids"])
#         print(output_ids)
#         print(masked_lm_positions)
#         print(masked_lm_ids)
#         sys.exit(-1)
        
        res = [
            np.array(
                output_ids, dtype=np.int64),
            np.array(
                encoded_inputs["token_type_ids"], dtype=np.int64),
            np.array(
                encoded_inputs["bbox"], dtype=np.int64),
            np.array(
                encoded_inputs["attention_mask"], dtype=np.int64),
            np.array(masked_lm_positions, dtype=np.int64),
            np.array(masked_lm_ids, dtype=np.int64),
            np.array(img, dtype=np.float32),
        ] 
#         print("===begin===")
#         for r in res:
#             print(r.shape)
#         print("===end===")
#         sys.exit(-1)
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
