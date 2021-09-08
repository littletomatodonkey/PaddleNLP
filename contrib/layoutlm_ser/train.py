import os
import sys
sys.path.insert(0, "../../")

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
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForTokenClassification

from xfun_dataset import XfunDatasetxForSer


def get_labels(path):
    labels = [
        "O",
        "b-question",
        "b-answer",
        "b-header",
        "i-answer",
        "i-question",
        "i-header",
    ]
    return labels


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def train(args):
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if paddle.distributed.get_rank() == 0 else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if paddle.distributed.get_rank() == 0 else logging.WARN, )

    labels = get_labels(args.labels)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    tokenizer = LayoutXLMTokenizer.from_pretrained(args.model_name_or_path)

    # for training process, model is needed for the bert class
    # else it can directly loaded for the downstream task
    if not args.do_train:
        model = LayoutXLMForTokenClassification.from_pretrained(
            args.model_name_or_path)
    else:
        model = LayoutXLMModel.from_pretrained(args.model_name_or_path)
        model = LayoutXLMForTokenClassification(
            model, num_classes=7, dropout=None)

    train_dataset = XfunDatasetxForSer(
        tokenizer,
        data_dir="./zh.train/",
        label_path="zh.train/xfun_normalize_train.json",
        img_size=(224, 224),
        pad_token_label_id=pad_token_label_id,
        add_special_ids=False)

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

    args.train_batch_size = args.per_gpu_train_batch_size * max(
        1, paddle.distributed.get_world_size())
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        use_shared_memory=True,
        collate_fn=None, )

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
            end_lr=args.learning_rate, )

    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay)

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
    tr_loss, logging_loss = 0.0, 0.0
    model.clear_gradients()
    train_iterator = trange(
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0])
    set_seed(
        args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(
            train_dataloader,
            desc="Iteration",
            disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                "input_ids": batch[0],
                "label_ids": batch[1],
                "token_type_ids": batch[2],
                "bbox": batch[3],
                "attention_mask": batch[4],
                "image": batch[5],
            }
            outputs = model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["label_ids"])
            # model outputs are always tuple in ppnlp (see doc)
            loss = outputs[0]

            loss = loss.mean()

            logger.info("train loss: {}".format(loss.numpy()))
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
                            labels,
                            pad_token_label_id,
                            mode="test", )
                        logger.info("results: {}".format(results))
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
             labels,
             pad_token_label_id,
             mode,
             prefix=""):
    eval_dataset = XfunDatasetxForSer(
        tokenizer,
        data_dir="./zh.val/",
        label_path="zh.val/xfun_normalize_val.json",
        img_size=(224, 224),
        pad_token_label_id=pad_token_label_id,
        add_special_ids=False)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(
        1, paddle.distributed.get_world_size())
    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        num_workers=8,
        use_shared_memory=True,
        collate_fn=None, )

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with paddle.no_grad():
            inputs = {
                "input_ids": batch[0],
                "label_ids": batch[1],
                "token_type_ids": batch[2],
                "bbox": batch[3],
                "attention_mask": batch[4],
                "image": batch[5],
            }

            outputs = model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["label_ids"])
            tmp_eval_loss, logits = outputs[:2]

            tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.numpy()
            out_label_ids = inputs["label_ids"].numpy()
        else:
            preds = np.append(preds, logits.numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["label_ids"].numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label.upper() for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
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

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
