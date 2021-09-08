import logging
import os

import paddle
from paddle.io import Dataset

logger = logging.getLogger(__name__)
import numpy as np

class FunsdDataset(Dataset):
    def __init__(self, args, tokenizer, labels, pad_token_label_id, mode):
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
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
        self.all_segment_ids = paddle.to_tensor(
            [f.segment_ids for f in features], dtype="int64")
        self.all_label_ids = paddle.to_tensor(
            [f.label_ids for f in features], dtype="int64")
        self.all_bboxes = paddle.to_tensor(
            [f.boxes for f in features], dtype="int64")
        self.image_seg_ids = paddle.to_tensor(
            [f.image_seg_ids for f in features], dtype="int64")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return (
            self.all_input_ids[index],
            self.all_input_mask[index],
            self.all_segment_ids[index],
            self.all_label_ids[index],
            self.all_bboxes[index],
            self.image_seg_ids[index])


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, boxes, actual_bboxes, file_name,
                 page_size):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids,
            input_mask,
            segment_ids,
            label_ids,
            boxes,
            actual_bboxes,
            file_name,
            page_size,
            image_seg_ids):
        assert (
            0 <= all(boxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            boxes)
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.boxes = boxes
        self.actual_bboxes = actual_bboxes
        self.file_name = file_name
        self.page_size = page_size
        self.image_seg_ids = image_seg_ids

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    box_file_path = os.path.join(data_dir, "{}_box.txt".format(mode))
    image_file_path = os.path.join(data_dir, "{}_image.txt".format(mode))
    guid_index = 1
    examples = []
    with open(
            file_path, encoding="utf-8") as f, open(
                box_file_path, encoding="utf-8") as fb, open(
                    image_file_path, encoding="utf-8") as fi:
        words = []
        boxes = []
        actual_bboxes = []
        file_name = None
        page_size = None
        labels = []
        for line, bline, iline in zip(f, fb, fi):
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(
                        InputExample(
                            guid="{}-{}".format(mode, guid_index),
                            words=words,
                            labels=labels,
                            boxes=boxes,
                            actual_bboxes=actual_bboxes,
                            file_name=file_name,
                            page_size=page_size,))
                    guid_index += 1
                    words = []
                    boxes = []
                    actual_bboxes = []
                    file_name = None
                    page_size = None
                    labels = []
            else:
                splits = line.split("\t")
                bsplits = bline.split("\t")
                isplits = iline.split("\t")
                assert len(splits) == 2
                assert len(bsplits) == 2
                assert len(isplits) == 4
                assert splits[0] == bsplits[0]
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[1].replace("\n", ""))
                    box = bsplits[-1].replace("\n", "")
                    box = [int(b) for b in box.split()]
                    boxes.append(box)
                    actual_bbox = [int(b) for b in isplits[1].split()]
                    actual_bboxes.append(actual_bbox)
                    page_size = [int(i) for i in isplits[2].split()]
                    file_name = isplits[3].strip()
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(
                InputExample(
                    guid="%s-%d".format(mode, guid_index),
                    words=words,
                    labels=labels,
                    boxes=boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,))
    return examples


def convert_examples_to_features(
        examples,
        label_list,
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

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        file_name = example.file_name
        page_size = example.page_size
        width, height = page_size
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens_doc = []
        token_boxes_doc = []
        actual_bboxes_doc = []
        label_ids_doc = []
        image_seg_ids_doc = []
        seg_no = 0
        for word, label, box, actual_bbox in zip(example.words, example.labels,
                                                 example.boxes,
                                                 example.actual_bboxes):
            word_tokens = tokenizer.tokenize(word)
            tokens_doc.extend(word_tokens)
            token_boxes_doc.extend([box] * len(word_tokens))
            actual_bboxes_doc.extend([actual_bbox] * len(word_tokens))
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            if label == "O":
                tokens_label = ["O"] * len(word_tokens)
            else:
                tokens_label = [f"I-{label}"] * len(word_tokens)
                tokens_label[0] = f"B-{label}"
            tokens_label_ids = [label_map[l] for l in tokens_label]
            label_ids_doc.extend(tokens_label_ids)
            image_seg_ids_doc.extend([ex_index, seg_no] * len(word_tokens))
            seg_no += 1
#         print(len(tokens_doc))
#         sys.exit(-1)

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        chunk_size = max_seq_length - special_tokens_count
        for chunk_id, index in enumerate(range(0, len(tokens_doc), chunk_size)):
            chunk_beg = index
            chunk_end = min(index + chunk_size, len(tokens_doc))
            tokens = tokens_doc[chunk_beg:chunk_end]
            token_boxes = token_boxes_doc[chunk_beg:chunk_end]
            actual_bboxes = actual_bboxes_doc[chunk_beg:chunk_end]
            label_ids = label_ids_doc[chunk_beg:chunk_end]
            chunk_beg = index * 2
            chunk_end = min((index + chunk_size) * 2, len(image_seg_ids_doc))
            image_seg_ids = image_seg_ids_doc[chunk_beg:chunk_end]
            tokens += [sep_token]
            token_boxes += [sep_token_box]
            actual_bboxes += [[0, 0, width, height]]
            label_ids += [pad_token_label_id]
            image_seg_ids += [-100, -100]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                token_boxes += [sep_token_box]
                actual_bboxes += [[0, 0, width, height]]
                label_ids += [pad_token_label_id]
                image_seg_ids += [-100, -100]
            segment_ids = [sequence_a_segment_id] * len(tokens) 

            if cls_token_at_end:
                tokens += [cls_token]
                token_boxes += [cls_token_box]
                actual_bboxes += [[0, 0, width, height]]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                image_seg_ids += [-100, -100]
            else:
                tokens = [cls_token] + tokens
                token_boxes = [cls_token_box] + token_boxes
                actual_bboxes = [[0, 0, width, height]] + actual_bboxes
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                image_seg_ids = [-100, -100] + image_seg_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

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
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                image_seg_ids = ([-100, -100] * padding_length) + image_seg_ids
                token_boxes = ([pad_token_box] * padding_length) + token_boxes
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                image_seg_ids += [-100, -100] * padding_length
                token_boxes += [pad_token_box] * padding_length
            
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(token_boxes) == max_seq_length
            assert len(image_seg_ids) == max_seq_length * 2
        
            if model_type != "layoutlm":
                input_mask = np.array(input_mask)
                input_mask = np.reshape(input_mask.astype(np.float32), 
                    [1, 1, input_mask.shape[0]])

#             input_mask = (1 - np.reshape(input_mask.astype(np.float32), 
#                 [1, 1, input_mask.shape[0]])) * -1e9

            #         if ex_index < 5:
            #             logger.info("*** Example ***")
            #             logger.info("guid: %s", example.guid)
            #             logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            #             logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            #             logger.info("input_mask: %s",
            #                         " ".join([str(x) for x in input_mask]))
            #             logger.info("segment_ids: %s",
            #                         " ".join([str(x) for x in segment_ids]))
            #             logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            #             logger.info("boxes: %s", " ".join([str(x) for x in token_boxes]))
            #             logger.info("actual_bboxes: %s",
            #                         " ".join([str(x) for x in actual_bboxes]))

            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    label_ids=label_ids,
                    boxes=token_boxes,
                    actual_bboxes=actual_bboxes,
                    file_name=file_name,
                    page_size=page_size,
                    image_seg_ids=image_seg_ids))
    return features
