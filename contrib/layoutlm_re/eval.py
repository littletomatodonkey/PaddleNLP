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
from metric import re_score

MODEL_CLASSES = {
    "layoutxlm":
    (LayoutXLMModel, LayoutXLMTokenizer, LayoutXLMForRelationExtraction),
    "layoutxlm-pp":
    (LayoutXLMPPModel, LayoutXLMPPTokenizer, LayoutXLMPPForRelationExtraction),
}


def cal_metric(re_preds, re_labels, entities):
    gt_relations = []
    for b in range(len(re_labels)):
        rel_sent = []
        for head, tail in zip(re_labels[b]["head"], re_labels[b]["tail"]):
            rel = {}
            rel["head_id"] = head
            rel["head"] = (entities[b]["start"][rel["head_id"]],
                           entities[b]["end"][rel["head_id"]])
            rel["head_type"] = entities[b]["label"][rel["head_id"]]

            rel["tail_id"] = tail
            rel["tail"] = (entities[b]["start"][rel["tail_id"]],
                           entities[b]["end"][rel["tail_id"]])
            rel["tail_type"] = entities[b]["label"][rel["tail_id"]]

            rel["type"] = 1
            rel_sent.append(rel)
        gt_relations.append(rel_sent)
    re_metrics = re_score(re_preds, gt_relations, mode="boundaries")
    return re_metrics


def evaluate(model, eval_dataloader, logger, prefix=""):
    # Eval!
    logger.info(f"***** Running evaluation {prefix} *****")
    logger.info(f"  Num examples = {len(eval_dataloader.dataset)}")

    re_preds = []
    re_labels = []
    entities = []
    eval_loss = 0.0
    model.eval()
    for idx, batch in enumerate(eval_dataloader):
        with paddle.no_grad():
            outputs = model(**batch)
            loss = outputs['loss'].mean().item()
            if paddle.distributed.get_rank() == 0:
                logger.info(
                    f"[Eval] process: {idx}/{len(eval_dataloader)}, loss: {loss:.5f}"
                )

            eval_loss += loss
        re_preds.extend(outputs['pred_relations'])
        re_labels.extend(batch['relations'])
        entities.extend(outputs['entities'])
    re_metrics = cal_metric(re_preds, re_labels, entities)
    re_metrics = {
        "precision": re_metrics["ALL"]["p"],
        "recall": re_metrics["ALL"]["r"],
        "f1": re_metrics["ALL"]["f1"],
    }
    model.train()
    return re_metrics


def eval(args):
    logger = get_logger()
    label2id_map, id2label_map = get_label_maps(args.label_map_path)
    pad_token_label_id = paddle.nn.CrossEntropyLoss().ignore_index

    base_model_class, tokenizer_class, model_class = MODEL_CLASSES[
        args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    model = model_class.from_pretrained(args.model_name_or_path)

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

    eval_dataloader = paddle.io.DataLoader(
        eval_dataset,
        batch_size=args.per_gpu_eval_batch_size,
        num_workers=8,
        shuffle=False,
        collate_fn=DataCollator())

    results = evaluate(model, eval_dataloader, logger)
    logger.info(f"eval results: {results}")


if __name__ == "__main__":
    args = parse_args()
    eval(args)
