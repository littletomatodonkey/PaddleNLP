import os
import random
import numpy as np
import paddle

from paddlenlp.transformers import LayoutXLMTokenizer, LayoutXLMModel, LayoutXLMForRelationExtraction
from paddlenlp.transformers import LayoutXLMPPTokenizer, LayoutXLMPPModel, LayoutXLMPPForRelationExtraction

from contrib.layoutlm_ser.xfun_dataset import XfunDataset
from contrib.layoutlm_ser.utils import parse_args, get_label_maps

from data_collator import DataCollator
from logger import get_logger
from eval import evaluate

MODEL_CLASSES = {
    "layoutxlm":
    (LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForRelationExtraction),
    "layoutxlm-pp":
    (LayoutXLMPPModel, LayoutXLMPPTokenizer, LayoutXLMPPForRelationExtraction),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def train(args):
    # Added here for reproducibility (even between python 2 and 3)
    set_seed(args.seed)

    logger = get_logger(log_file=os.path.join(args.output_dir, "train.log"))

    label2id_map, id2label_map = get_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    base_model_class, tokenizer_class, model_class = MODEL_CLASSES[
        args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    model = base_model_class.from_pretrained(args.model_name_or_path)
    model = model_class(model, dropout=None)

    # dist mode
    if paddle.distributed.get_world_size() > 1:
        model = paddle.distributed.DataParallel(model)

    train_dataset = XfunDataset(
        tokenizer,
        data_dir=args.train_data_dir,
        label_path=args.train_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        max_seq_len=args.max_seq_length,
        pad_token_label_id=pad_token_label_id,
        contains_re=True,
        add_special_ids=False,
        return_attention_mask=True,
        model_type=args.model_type,
        load_mode='all')

    eval_dataset = XfunDataset(
        tokenizer,
        data_dir=args.eval_data_dir,
        label_path=args.eval_label_path,
        label2id_map=label2id_map,
        img_size=(224, 224),
        max_seq_len=args.max_seq_length,
        pad_token_label_id=pad_token_label_id,
        contains_re=True,
        add_special_ids=False,
        return_attention_mask=True,
        model_type=args.model_type,
        load_mode='all')

    train_sampler = paddle.io.DistributedBatchSampler(
        train_dataset, batch_size=args.per_gpu_train_batch_size, shuffle=True)
    args.train_batch_size = args.per_gpu_train_batch_size * \
                            max(1, paddle.distributed.get_world_size())
    train_dataloader = paddle.io.DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=8,
        use_shared_memory=True,
        collate_fn=DataCollator())

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=DataCollator())

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
            end_lr=args.learning_rate, )
    grad_clip = paddle.nn.ClipGradByNorm(clip_norm=10)
    optimizer = paddle.optimizer.Adam(
        learning_rate=args.learning_rate,
        parameters=model.parameters(),
        epsilon=args.adam_epsilon,
        grad_clip=grad_clip,
        weight_decay=args.weight_decay)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {args.train_batch_size * paddle.distributed.get_world_size()}"
    )
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0
    model.clear_gradients()
    train_dataloader_len = len(train_dataloader)
    best_metirc = {'f1': 0}
    model.train()

    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            # model outputs are always tuple in ppnlp (see doc)
            loss = outputs['loss']
            loss = loss.mean()

            logger.info(
                f"epoch: [{epoch}/{args.num_train_epochs}], iter: [{step}/{train_dataloader_len}], global_step:{global_step}, train loss: {np.mean(loss.numpy())}, lr: {optimizer.get_lr()}"
            )

            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            # lr_scheduler.step()  # Update learning rate schedule

            global_step += 1

            if (paddle.distributed.get_rank() == 0 and args.eval_steps > 0 and
                    global_step % args.eval_steps == 0):
                # Log metrics
                if (paddle.distributed.get_rank() == 0 and args.
                        evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    results = evaluate(model, eval_dataloader, logger)
                    if results['f1'] > best_metirc['f1']:
                        best_metirc = results
                        output_dir = os.path.join(args.output_dir,
                                                  "checkpoint-best")
                        os.makedirs(output_dir, exist_ok=True)
                        model.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)
                        paddle.save(args,
                                    os.path.join(output_dir,
                                                 "training_args.bin"))
                        logger.info(f"Saving model checkpoint to {output_dir}")
                    logger.info(f"eval results: {results}")
                    logger.info(f"best_metirc: {best_metirc}")

            if (paddle.distributed.get_rank() == 0 and args.save_steps > 0 and
                    global_step % args.save_steps == 0):
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "checkpoint-latest")
                os.makedirs(output_dir, exist_ok=True)
                if paddle.distributed.get_rank() == 0:
                    model.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    paddle.save(args,
                                os.path.join(output_dir, "training_args.bin"))
                    logger.info(f"Saving model checkpoint to {output_dir}")
    logger.info(f"best_metirc: {best_metirc}")


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
