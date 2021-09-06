import logging
import os, sys

# import paddle
# from paddle.io import Dataset
import random

logger = logging.getLogger(__name__)
import numpy as np
import h5py
import cv2

import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])
def read_file_path_list(input_path):
    file_paths = []
    if os.path.isfile(input_path):
        file_paths.append(input_path)
    else:
        for root, _, fs in os.walk(input_path):
            for f in fs:
                file_paths.append(os.path.join(root, f))
    return file_paths

def get_texts_info(ocr_res_path):
    words = []
    boxes = []
    with open(ocr_res_path, "rb") as fin:
        lines = fin.readlines()
        bbox_num = len(lines)
        for bno in range(bbox_num):
            line = lines[bno].decode('utf-8').strip("\n").split(",")
            bbox_4pts = [float(x) for x in line[0:8]]
            left = int(min(bbox_4pts[::2]))
            right = int(max(bbox_4pts[::2]))
            top = int(min(bbox_4pts[1::2]))
            bottom = int(max(bbox_4pts[1::2]))
            bbox = [left, top, right, bottom]
            text = line[8]
            words.append(text)
            boxes.append(bbox)
    return words, boxes

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            boxes,
            masked_lm_positions,
            masked_lm_ids):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(boxes)
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.boxes = boxes
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids
        
def convert_words_bboxes_to_features(
    words,
    bboxes,
    tokenizer,
    max_seq_length=512,
    max_predictions_per_seq = 20,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    cls_token_box=[0, 0, 0, 0],
    sep_token_box=[1000, 1000, 1000, 1000],
    pad_token_box=[0, 0, 0, 0],
    pad_token_segment_id=0,
    pad_token_label_id=-1,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    model_type="bert"):
    
    features = []
    tokens_doc = []
    token_boxes_doc = []
    for word, box in zip(words, bboxes):
        word_tokens = tokenizer.tokenize(word)
        tokens_doc.extend(word_tokens)
        token_boxes_doc.extend([box] * len(word_tokens))

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2
    chunk_size = max_seq_length - special_tokens_count
    for chunk_id, index in enumerate(range(0, len(tokens_doc), chunk_size)):
        chunk_beg = index
        chunk_end = min(index + chunk_size, len(tokens_doc))
        tokens = tokens_doc[chunk_beg:chunk_end]
        token_boxes = token_boxes_doc[chunk_beg:chunk_end]
        tokens += [sep_token]
        token_boxes += [sep_token_box]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            token_boxes += [sep_token_box]
        segment_ids = [sequence_a_segment_id] * len(tokens) 

        if cls_token_at_end:
            tokens += [cls_token]
            token_boxes += [cls_token_box]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            token_boxes = [cls_token_box] + token_boxes
            segment_ids = [cls_token_segment_id] + segment_ids

        #####create_masked_lm_predictions
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)
        random.shuffle(cand_indexes)
        output_tokens = list(tokens)
        masked_lm_prob = 0.15
        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 20% of the time, keep original
                masked_token = tokens[index]
            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        masked_lm_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)

        #####end create_masked_lm_predictions

#             input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length
                          ) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length
                           ) + segment_ids
            token_boxes = ([pad_token_box] * padding_length) + token_boxes
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            token_boxes += [pad_token_box] * padding_length

        assert len(input_mask) == max_seq_length
        assert len(input_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(token_boxes) == max_seq_length

        if model_type != "layoutlm":
            input_mask = np.array(input_mask)
            input_mask = np.reshape(input_mask.astype(np.float32), 
                [1, 1, input_mask.shape[0]])

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                boxes=token_boxes,
                masked_lm_positions=masked_lm_positions,
                masked_lm_ids=masked_lm_ids))
    return features

def convert_features_to_numpy(all_features):
    num_instances = len(all_features)
    all_features_np = {}
    all_features_np["input_ids"] = np.array(
        [f.input_ids for f in all_features], dtype="int32")
    all_features_np["input_mask"] = np.concatenate(
        [f.input_mask for f in all_features])
    all_features_np["boxes"] = np.array(
        [f.boxes for f in all_features], dtype="int32")
    all_features_np["masked_lm_positions"] = np.array(
        [f.masked_lm_positions for f in all_features], dtype="int32")
    all_features_np["masked_lm_ids"] = np.array(
        [f.masked_lm_ids for f in all_features], dtype="int32")
    return all_features_np

def write_features_np_to_h5py(output_file, features):
    print("saving data")
    f = h5py.File(output_file, 'w')
    f.create_dataset(
        "input_ids", data=features["input_ids"], dtype='i4', compression='gzip')
    f.create_dataset(
        "input_mask",
        data=features["input_mask"],
        dtype='i4',
        compression='gzip')
    f.create_dataset(
        "boxes",
        data=features["boxes"],
        dtype='i4',
        compression='gzip')
    f.create_dataset(
        "masked_lm_positions",
        data=features["masked_lm_positions"],
        dtype='i4',
        compression='gzip')
    f.create_dataset(
        "masked_lm_ids",
        data=features["masked_lm_ids"],
        dtype='i4',
        compression='gzip')
    f.flush()
    f.close()    
    
def read_features_h5py_to_np(input_file):
    f = h5py.File(input_file, "r")
    keys = ['input_ids', 'input_mask', 'boxes', 'masked_lm_positions', 'masked_lm_ids']
    inputs = [np.asarray(f[key][:]) for key in keys]
    return inputs

sys.path.insert(0, "/paddle/lcode/gry/PaddleNLP/")
from paddlenlp.transformers import ErnieGramTokenizer
tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')

def gen_nlpbert_format(total_num=12, cur_no=0):
    set_path = "./data/yanbao_v1/"
    output_file = set_path + "train_data/yanbao_v1_train.h5.%d" % (cur_no)
    ocr_res_set_path = set_path + "result/"
    max_norm_size = 1000
    max_seq_length = 512
    max_predictions_per_seq = 20
    file_paths = read_file_path_list(ocr_res_set_path)
    count = 0
    all_features = []
    for fno in range(len(file_paths)):
        ocr_res_path = file_paths[fno]
        if 'DS_Store' in ocr_res_path:
            continue
        if fno % total_num != cur_no:
            continue
        if (fno - cur_no) / total_num % 100 == 0:
            print(cur_no, fno, len(file_paths), len(all_features))
#         if fno > 1000:
#             break
        img_path = ocr_res_path.replace('result', '研报JPEG')
        img_path = img_path.replace('.txt', '')
        img = cv2.imread(img_path)
        if img is None:
            print("loading error:", ocr_res_path, img_path)
            continue
        img_height, img_width = img.shape[0:2]
        words, boxes = get_texts_info(ocr_res_path)
        features = convert_words_bboxes_to_features(
            words, boxes, tokenizer, max_seq_length, max_predictions_per_seq)
        for f in features:
            all_features.append(f)
    all_features_np = convert_features_to_numpy(all_features)
    write_features_np_to_h5py(output_file, all_features_np)
    print("Done:", output_file, cur_no, total_num)

def merge_h5py():
    input_files_set = "./data/yanbao_v1/train_data/"
    input_files = os.listdir(input_files_set)
    inputs_all = []
    for input_file in input_files:
        inputs = read_features_h5py_to_np(input_files_set + input_file)
        inputs_all.append(inputs)
    all_features_np = {}
    all_features_np["input_ids"] = np.concatenate([f[0] for f in inputs_all])
    all_features_np["input_mask"] = np.concatenate([f[1] for f in inputs_all])
    all_features_np["boxes"] = np.concatenate([f[2] for f in inputs_all])
    all_features_np["masked_lm_positions"] = np.concatenate([f[3] for f in inputs_all])
    all_features_np["masked_lm_ids"] = np.concatenate([f[4] for f in inputs_all])
    output_file = "./data/yanbao_v1/yanbao_v1_train.h5"
    write_features_np_to_h5py(output_file, all_features_np)
    inputs = read_features_h5py_to_np(output_file)
    print(len(inputs))
    for tmp in inputs:
        print(tmp.shape)
    print("ok")

if __name__ == "__main__":
    method_type = sys.argv[1]
    total_num = int(sys.argv[2])
    cur_no = int(sys.argv[3]) - 1
    if method_type == "gen":
        gen_nlpbert_format(total_num=total_num, cur_no=cur_no)
    else:
        merge_h5py()
    
    