import json
import random
import os
import sys
import cv2
import numpy as np
import paddle
from paddle.io import Dataset, DataLoader
from paddlenlp.data import Stack

__all__ = ["PretrainingDataset"]

import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

        
def _collate_data(data, stack_fn=Stack()):
    num_fields = len(data[0])
    out = [None] * num_fields
    # input_ids, token_type_ids, bbox, attention_mask, masked_lm_positions, masked_lm_ids, img
    for i in (0, 1, 2, 3, 6):
        out[i] = stack_fn([x[i] for x in data])

    batch_size, seq_length = out[0].shape
    size = sum(len(x[4]) for x in data)
#     # Padding for divisibility by 8 for fp16 or int8 usage
#     if size % 8 != 0:
#         size += 8 - (size % 8)
    # masked_lm_positions
    # Organize as a 1D tensor for gather or use gather_nd
    out[4] = np.full(size, 0, dtype=np.int32)
    # masked_lm_labels
    out[5] = np.full([size, 1], -1, dtype=np.int64)
    mask_token_num = 0
    for i, x in enumerate(data):
        for j, pos in enumerate(x[4]):
#             out[4][mask_token_num] = i * seq_length + pos
            out[4][mask_token_num] = i * (seq_length + 49) + pos
            out[5][mask_token_num] = x[5][j]
            mask_token_num += 1
    # mask_token_num
#     out.append(np.asarray([mask_token_num], dtype=np.float32))
#     out[5] = stack_fn(data[5])
    return out


class PretrainingDataset(Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            label_path,
            max_seq_length=512, 
            img_size=(224, 224),
            pad_token_label_id=None,
            return_attention_mask=True,
            model_type='layoutxlm'):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_path = label_path
        self.max_seq_length = max_seq_length
        self.pad_token_label_id = pad_token_label_id
        self.return_attention_mask = return_attention_mask
        if self.pad_token_label_id is None:
            self.pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
        self.all_lines = self.read_all_lines()
        self.model_type = model_type
        
    def read_all_lines(self, ):
        with open(self.label_path, "r") as fin:
            lines = fin.readlines()
        return lines
    
    def __len__(self, ):
        return len(self.all_lines)

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

#     def truncate_inputs(self, encoded_inputs, max_seq_len=512):
#         for key in encoded_inputs:
#             length = min(len(encoded_inputs[key]), max_seq_len)
#             encoded_inputs[key] = encoded_inputs[key][:length]
#         return encoded_inputs
    
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

    
    def parse_label_file(self, line):
        info_dict = json.loads(line)
        image_name = info_dict['img_path']
        image_path = os.path.join(self.data_dir, image_name)
#         print(image_path)
        
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
    
        height = info_dict['shape'][0]
        width = info_dict['shape'][1]

        words_list = []
        bbox_list = []
        input_ids_list = []
        token_type_ids_list = []
        masked_lm_positions_list = []
        masked_lm_ids_list = []

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
#             print("text: ", len(text), text)
#             print("input_ids: ", len(encode_res['input_ids']), encode_res['input_ids'])
#             print("token_type_ids: ", len(encode_res['token_type_ids']), encode_res['token_type_ids'])
#             print("bbox: ", bbox)
            
#             tokens = self.tokenizer.tokenize(text)
#             print("tokens: ", len(tokens), tokens)
#             print('\n')
# #             exit(-1)
            
            input_ids_list.extend(encode_res['input_ids'])
            token_type_ids_list.extend(encode_res["token_type_ids"])
            bbox_list.extend([bbox] * len(encode_res["input_ids"]))

        encoded_inputs = {
            "input_ids": input_ids_list,
            "token_type_ids": token_type_ids_list,
            "bbox": bbox_list,
            "attention_mask": [1] * len(input_ids_list),
        }
#         print(encoded_inputs)
#         exit(-1)

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
#         ss = ""
#         for i in range(len(output_ids)):
#             ss += self.tokenizer.convert_ids_to_tokens(output_ids[i])
#         print("original tokens: ", ss)
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
#             print(self.tokenizer.convert_ids_to_tokens(p.label))
            masked_lm_positions.append(p.index)
            masked_lm_ids.append(p.label)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)

#         ss = ""
#         for i in range(len(output_ids)):
#             ss += self.tokenizer.convert_ids_to_tokens(output_ids[i])
#         print("masked tokens: ", ss)
        
#         print("token_type_ids: ", encoded_inputs["token_type_ids"])
#         print("attention_mask: ", encoded_inputs["attention_mask"])
#         print("masked_lm_positions: ", masked_lm_positions)
#         print("masked_lm_ids: ", masked_lm_ids)
#         for i in range(len(masked_lm_positions)):
#             print(self.tokenizer.convert_ids_to_tokens(output_ids[masked_lm_positions[i]]), self.tokenizer.convert_ids_to_tokens(masked_lm_ids[i]))
#         exit(-1)
        
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

        return res

    
    def __getitem__(self, idx):
        res = self.parse_label_file(self.all_lines[idx])
#         print("===begin===")
#         for r in res:
#             print(type(r), r.shape)
#         print("===end===")
        return res
