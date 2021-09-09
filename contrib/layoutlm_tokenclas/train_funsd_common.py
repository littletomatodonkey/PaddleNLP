import os
import sys
sys.path.insert(0, "/paddle/lcode/gry/PaddleNLP/")

import paddle

import numpy as np
import random

from tqdm import tqdm, trange

import logging
logger = logging.getLogger(__name__)

from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score, )

# relative reference
from utils import parse_args
from funsd_common import FunsdDataset
# from paddlenlp.transformers import LayoutLMModel, LayoutLMForTokenClassification, LayoutLMTokenizer

from paddlenlp.transformers import BertForTokenClassification, BertTokenizer
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer
from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer
from paddlenlp.transformers import ErnieGramForTokenClassification, ErnieGramTokenizer
from paddlenlp.transformers import NeZhaForTokenClassification, NeZhaTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
import paddle.distributed as dist

MODEL_CLASSES = {
    "bert":(BertForTokenClassification, BertTokenizer),
    "ernie":(ErnieForTokenClassification, ErnieTokenizer),
    "roberta":(RobertaForTokenClassification, RobertaTokenizer),
    "erniegram":(ErnieGramForTokenClassification, ErnieGramTokenizer),
    "nezha":(NeZhaForTokenClassification, NeZhaTokenizer)
}

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def train(args):
    device = 'gpu:{}'.format(dist.ParallelEnv().dev_id)
    device = paddle.set_device(device)
    distributed = dist.get_world_size() != 1
    if distributed:
        dist.init_parallel_env()
    
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if paddle.distributed.get_rank() == 0 else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if paddle.distributed.get_rank() == 0 else logging.WARN, )

    labels = get_labels(args.labels)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index
    
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    # for training process, model is needed for the bert class
    # else it can directly loaded for the downstream task
    model = model_class.from_pretrained(args.model_name_or_path, num_classes=len(labels))
    if distributed:
        model = paddle.DataParallel(model)
    loss_fct = paddle.nn.loss.CrossEntropyLoss()
    
    train_dataset = FunsdDataset(
        args, tokenizer, labels, pad_token_label_id, mode="train")

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

    args.train_batch_size = args.per_gpu_train_batch_size * max(
        1, paddle.distributed.get_world_size())

    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=None)

    t_total = len(train_dataloader
                  ) // args.gradient_accumulation_steps * args.num_train_epochs

    # build linear decay with warmup lr sch
    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(
        learning_rate=args.learning_rate,
        decay_steps=t_total,
        end_lr=0.0,
        power=1.0)

    if args.warmup_steps > 0:
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(
            lr_scheduler,
            args.warmup_steps,
            start_lr=0,
            end_lr=args.learning_rate)
    
#     optimizer = paddle.optimizer.AdamW(
#         learning_rate=lr_scheduler,
#         parameters=model.parameters(),
#         epsilon=args.adam_epsilon,
#         weight_decay=args.weight_decay)
    
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        beta1=0.9,
        beta2=0.999,
        epsilon=args.adam_epsilon,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)    

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size * paddle.distributed.get_world_size(), )
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epoch_num = 0
    best_info = {}
    tr_loss, logging_loss = 0.0, 0.0
    model.clear_gradients()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0])
    set_seed(
        args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_num += 1
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            # model.eval()
            model.train()
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1]
            }
#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "labels": batch[3],
#             }
#             if args.model_type in ["layoutlm"]:
#                 inputs["bbox"] = batch[4]
#             inputs["token_type_ids"] = (
#                 batch[2] if args.model_type in ["bert", "layoutlm"] else
#                 None)  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)

            loss = loss_fct(outputs, batch[3])
            loss = loss.mean()

            logger.info("gstep_epoch:{} {} train loss: {}".format(
                global_step, epoch_num, loss.numpy()))
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()  # Update learning rate schedule
                model.clear_gradients()
                global_step += 1

                if (paddle.distributed.get_rank() == 0 and
                        args.logging_steps > 0 and
                        global_step % args.logging_steps == 0):
                    # Log metrics
                    if (paddle.distributed.get_rank() == 0 and args.
                            evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(
                            args,
                            model,
                            tokenizer,
                            loss_fct,
                            labels,
                            pad_token_label_id,
                            mode="test", )
                        logger.info("gstep_epoch:{} {} results: {}".format(
                            global_step, epoch_num, results))
                        if 'result' not in best_info or best_info['result'][2]['f1'] < results['f1']:
                            best_info['result'] = [global_step, epoch_num, results]
                        best_global_step, best_epoch_num, best_results = best_info['result']
                        logger.info("best_info: gstep_epoch:{} {} results: {}".format(best_global_step, best_epoch_num, best_results))
                    logging_loss = tr_loss

                if (args.local_rank in [-1, 0] and args.save_steps > 0 and
                        global_step % args.save_steps == 0):
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, "checkpoint-{}".format(global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    if paddle.distributed.get_rank() == 0:
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(
                            args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def evaluate(args,
             model,
             tokenizer,
             loss_fct,
             labels,
             pad_token_label_id,
             mode,
             prefix=""):
    eval_dataset = FunsdDataset(
        args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(
        1, paddle.distributed.get_world_size())
    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=None, )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    image_seg_ids_all = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with paddle.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
            }
            inputs_labels = batch[3]
            image_seg_ids = batch[5]
            image_seg_ids = image_seg_ids.numpy()
            if image_seg_ids_all is None:
                image_seg_ids_all = image_seg_ids
            else:
                image_seg_ids_all = np.append(image_seg_ids_all, image_seg_ids, axis=0)

#             inputs = {
#                 "input_ids": batch[0],
#                 "attention_mask": batch[1],
#                 "labels": batch[3],
#             }
            if args.model_type in ["layoutlm"]:
                inputs["bbox"] = batch[4]
#             inputs["token_type_ids"] = (
#                 batch[2] if args.model_type in ["bert", "layoutlm"] else
#                 None)  # RoBERTa don"t use segment_ids
            logits = model(**inputs)
            tmp_eval_loss = loss_fct(logits, batch[3])
#             outputs = model(**inputs)
#             tmp_eval_loss, logits = outputs[:2]

            tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.numpy()
            out_label_ids = inputs_labels.numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs_labels.numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i][j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    ###get segments results
    image_seg_ids_map = {}
    for i in range(image_seg_ids_all.shape[0]):
        for j in range(0, image_seg_ids_all.shape[1], 2):
            img_id = image_seg_ids_all[i][j]
            seg_id = image_seg_ids_all[i][j+1]
            if img_id >=0 and seg_id >=0:
                key = "%d_%d" % (img_id, seg_id)
                if key not in image_seg_ids_map:
                    image_seg_ids_map[key] = []
                image_seg_ids_map[key].append((i, int(j/2)))
                
    dt_num = 0
    gt_num = 0
    same_num = 0
    seg_out_label_list = []
    seg_preds_list = []
    for key in image_seg_ids_map:
        tmp_labels_dict = {}
        tmp_preds_dict = {}
        for (i, j) in image_seg_ids_map[key]:
            tmp_label = label_map[out_label_ids[i][j]].strip("I-")
            tmp_pred = label_map[preds[i][j]].strip("I-")
            tmp_label = tmp_label.strip("B-")
            tmp_pred = tmp_pred.strip("B-")
            if tmp_label not in tmp_labels_dict:
                tmp_labels_dict[tmp_label] = 0
            if tmp_pred not in tmp_preds_dict:
                tmp_preds_dict[tmp_pred] = 0
            tmp_labels_dict[tmp_label] += 1
            tmp_preds_dict[tmp_pred] += 1
        max_label = sorted(tmp_labels_dict.items(),
                           key=lambda e:e[1], reverse=True)[0][0]
        max_pred = sorted(tmp_preds_dict.items(),
                           key=lambda e:e[1], reverse=True)[0][0]
#         seg_out_label_list.append([max_label])
#         seg_preds_list.append([max_pred])
        if max_label == "O":
            continue
        dt_num += 1
        gt_num += 1
        if max_label == max_pred:
            same_num += 1
    
    precision_seg = same_num * 1.0 / dt_num
    recall_seg = same_num * 1.0 / gt_num
    f1_seg = 2 * (precision_seg * recall_seg) / (precision_seg + recall_seg)

#     results = {
#         "loss": eval_loss,
#         "precision": precision_score(out_label_list, preds_list),
#         "recall": recall_score(out_label_list, preds_list),
#         "f1": f1_score(out_label_list, preds_list),
#         "precision_seg": precision_score(seg_out_label_list, seg_preds_list),
#         "recall_seg": recall_score(seg_out_label_list, seg_preds_list),
#         "f1_seg": f1_score(seg_out_label_list, seg_preds_list)
#     }
    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "precision_seg": precision_seg,
        "recall_seg": recall_seg,
        "f1_seg": f1_seg
    }

    with open("test_gt.txt", "w") as fout:
        for lbl in out_label_list:
            for l in lbl:
                fout.write(l + "\t")
            fout.write("\n")
    with open("test_pred.txt", "w") as fout:
        for lbl in preds_list:
            for l in lbl:
                fout.write(l + "\t")
            fout.write("\n")

    report = classification_report(out_label_list, preds_list)
    logger.info("\n" + report)
#     report = classification_report(seg_out_label_list, seg_preds_list)
#     logger.info("\n" + report)

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
        
    return results, preds_list


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
