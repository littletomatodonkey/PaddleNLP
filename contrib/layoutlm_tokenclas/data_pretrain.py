import logging
import os

import paddle
from paddle.io import Dataset
import random

logger = logging.getLogger(__name__)
import numpy as np

import collections
MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

from create_pretrain_data import read_features_h5py_to_np

class DatasetForPretrainH5py(Dataset):
    def __init__(self, args):
        logger.info("Creating features from dataset file at %s", args.data_dir)
        self.inputs = read_features_h5py_to_np(args.data_dir)
        
    def __len__(self):
        return len(self.inputs[0])

    def __getitem__(self, index):
#         input_ids = paddle.to_tensor(self.inputs[0][index], dtype="int64")
#         input_mask = paddle.to_tensor(self.inputs[1][index], dtype="int64")
#         boxes = paddle.to_tensor(self.inputs[2][index], dtype="int64")
#         masked_lm_positions = paddle.to_tensor(self.inputs[3][index], dtype="int64")
#         masked_lm_ids = paddle.to_tensor(self.inputs[4][index], dtype="int64")
        input_ids = self.inputs[0][index]
        input_mask = self.inputs[1][index].reshape(1, 1, -1)
        boxes = self.inputs[2][index]
        masked_lm_positions = self.inputs[3][index]
        masked_lm_ids = self.inputs[4][index]
        return (input_ids, input_mask, masked_lm_positions, masked_lm_ids, boxes)
    
class DatasetForPretrain(Dataset):
    def __init__(self, args, tokenizer, pad_token_label_id, mode):
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            args.max_seq_length,
            tokenizer,
            cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_label_id=pad_token_label_id,
            model_type=args.model_type)

        self.features = features
        # Convert to Tensors and build dataset
        self.all_input_ids = paddle.to_tensor(
            [f.input_ids for f in features], dtype="int64")
        self.all_input_mask = paddle.to_tensor(
            [f.input_mask for f in features], dtype="int64")
        self.all_masked_lm_positions = paddle.to_tensor(
            [f.masked_lm_positions for f in features], dtype="int64")
        self.all_masked_lm_ids = paddle.to_tensor(
            [f.masked_lm_ids for f in features], dtype="int64")
        self.all_bboxes = paddle.to_tensor(
            [f.boxes for f in features], dtype="int64")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_masked_lm_positions[index],
            self.all_masked_lm_ids[index],
            self.all_bboxes[index])


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, boxes):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.boxes = boxes

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

def read_examples_from_file(data_dir, mode):
#     box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    box_file_path = os.path.join(data_dir)
    guid_index = 1
    examples = []
    with open(box_file_path, encoding="utf-8") as fb:
        words = []
        boxes = []
        for bline in fb:
            if bline.startswith("-DOCSTART-") or bline == "" or bline == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            boxes=boxes))
                    guid_index += 1
                    words = []
                    boxes = []
            else:
                bsplits = bline.split("\t")
                assert len(bsplits) == 2
                words.append(bsplits[0])
                box = bsplits[-1].replace("\n", "")
                box = [int(b) for b in box.split()]
                boxes.append(box)
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    boxes=boxes))
    return examples

def convert_examples_to_features(
        examples,
        max_seq_length,
        tokenizer,
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
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens_doc = []
        token_boxes_doc = []
        for word, box in zip(example.words, example.boxes):
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
            max_predictions_per_seq = 20
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
