import os
import sys
sys.path.insert(0, "../../")

import paddle

import numpy as np
import random
import copy
import logging
logger = logging.getLogger(__name__)

# relative reference
from utils import parse_args
from paddlenlp.transformers import LayoutXLMModel, LayoutXLMForPretraining, LayoutXLMTokenizer
from paddlenlp.transformers import LayoutXLMPPModel, LayoutXLMPPForPretraining, LayoutXLMPPTokenizer

from pretrain_dataset import PretrainingDataset, _collate_data

import paddle.distributed as dist

MODEL_CLASSES = {
    "layoutxlm": (LayoutXLMModel, LayoutXLMForPretraining, LayoutXLMTokenizer),
    "layoutxlm-pp": (LayoutXLMPPModel, LayoutXLMPPForPretraining, LayoutXLMPPTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    paddle.seed(args.seed)


def train(args):        
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if paddle.distributed.get_rank() == 0 else None,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if paddle.distributed.get_rank() == 0 else logging.WARN, )

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    
    # dist mode
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()
        
    args.model_type = args.model_type.lower()
    base_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    
    pretrained_models_list = list(
        model_class.pretrained_init_configuration.keys())
        
    model = base_class.from_pretrained(args.model_name_or_path)
    model = model_class(model)
    
    # dist mode
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    loss_fct = paddle.nn.loss.CrossEntropyLoss()
        
    train_dataset = PretrainingDataset(
        tokenizer,
        data_dir=args.train_data_dir,
        label_path=args.train_label_path,
        max_seq_length=args.max_seq_length,
        img_size=(224, 224))
    
    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)

    args.train_batch_size = args.per_gpu_train_batch_size * max(
        1, paddle.distributed.get_world_size())
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        use_shared_memory=True,
        collate_fn=_collate_data,
        return_list=True)
    
    t_total = len(train_dataloader) * args.num_train_epochs

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
        "  Total train batch size (w. parallel, distributed) = %d",
        args.train_batch_size * paddle.distributed.get_world_size(), )
    logger.info("  Total optimization steps = %d", t_total)

    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.clear_gradients()
    set_seed(args)
    
    for epoch_id in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {
                "input_ids": batch[0],
                "token_type_ids": batch[1],
                "bbox": batch[2],
                "attention_mask": batch[3],
                "masked_positions": batch[4],
                "image": batch[6],
            }
            outputs = model(
                input_ids=inputs["input_ids"],
                bbox=inputs["bbox"],
                image=inputs["image"],
                token_type_ids=inputs["token_type_ids"],
                attention_mask=inputs["attention_mask"],
                masked_positions=inputs["masked_positions"])

            loss = loss_fct(outputs, batch[5])
            loss = loss.mean()

            logger.info("[epoch {}/{}][iter: {}/{}] lr: {}, train loss: {}, ".
                        format(epoch_id, args.num_train_epochs, step,
                               len(train_dataloader),
                               lr_scheduler.get_lr(), loss.numpy()[0]))

            loss.backward()

            tr_loss += loss.item()
            optimizer.step()
            lr_scheduler.step()  # Update learning rate schedule
            model.clear_gradients()
            global_step += 1

            if paddle.distributed.get_rank(
            ) == 0 and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir,
                                          "checkpoint-{}".format(global_step))
                os.makedirs(output_dir, exist_ok=True)
                if paddle.distributed.get_rank() == 0:
                    if paddle.distributed.get_world_size() > 1:
                        model._layers.save_pretrained(output_dir)
                    else:
                        model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
    return global_step, tr_loss / global_step


if __name__ == "__main__":
    args = parse_args()
    train(args)
