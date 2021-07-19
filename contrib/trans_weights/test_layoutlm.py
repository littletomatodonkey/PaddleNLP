import sys
import os
import paddle
import torch
import numpy as np

sys.path.insert(0, "./layoutlm_paddle/PaddleNLP")
# sys.path.insert(0, "./layoutlm_torch")

import transformers

torch_model_dir = "microsoft/layoutlm-base-uncased/"


def get_input_demo():
    words = ["train data", "eval data"]
    normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
    # words = ["train data", ]
    # normalized_word_boxes = [[637, 773, 693, 782], ]
    return words, normalized_word_boxes


def preprcess_data_paddle(x_list):
    from paddlenlp.data import Tuple, Pad
    if isinstance(x_list, dict):
        x_list = [x_list]
    y_dict = dict()
    for x_dict in x_list:
        for key in x_dict:
            val = x_dict[key]
            if key not in y_dict:
                y_dict[key] = []
            y_dict[key].append(val)
    pad_func = Pad(axis=0, pad_val=0)
    # already contains input_ids, token_type_ids
    # we need to append attention_mask in it
    for key in y_dict:
        y_dict[key] = pad_func(y_dict[key])

    return y_dict


def test_layoutlm_paddle():
    from paddlenlp.transformers import LayoutLMModel, LayoutLMTokenizer

    tokenizer = LayoutLMTokenizer.from_pretrained(
        "./layoutlm-base-uncased-paddle//")
    model = LayoutLMModel.from_pretrained("./layoutlm-base-uncased-paddle//")
    model.eval()

    # tokenizer.save_pretrained("./test")
    # model.save_pretrained("./test")

    words, normalized_word_boxes = get_input_demo()

    tokenizer_res = tokenizer(
        ' '.join(words),
        pad_to_max_seq_len=False,
        truncation_strategy="longest_first",
        max_seq_len=100,
        return_attention_mask=True)
    print(tokenizer_res)

    token_boxes = []
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    tokenizer_res = preprcess_data_paddle(tokenizer_res)

    input_ids = paddle.to_tensor(tokenizer_res["input_ids"])
    token_boxes = paddle.to_tensor([token_boxes])
    print(
        f"input_ids shape: {input_ids.shape}, token_boxes shape: {token_boxes.shape}"
    )

    output = model(input_ids, bbox=token_boxes)

    # last_hidden_state
    print(output[0].shape)  # batch_size x 7 x 768
    # pooler_output batch_size x 768
    print(output[1].shape)

    return output[0], output[1]


def test_layoutlm_torch():
    from transformers import LayoutLMTokenizer, LayoutLMModel

    tokenizer = LayoutLMTokenizer.from_pretrained(
        "microsoft/layoutlm-base-uncased/")
    model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased/")
    model.eval()
    words, normalized_word_boxes = get_input_demo()
    print(f"[input] words: f{words}, boxes: f{normalized_word_boxes}")

    token_boxes = []
    # exit()
    for word, box in zip(words, normalized_word_boxes):
        word_tokens = tokenizer.tokenize(word)
        # print(len(word_tokens))
        # print(word_tokens)
        token_boxes.extend([box] * len(word_tokens))
    print(f"token_boxes: f{token_boxes}")
    # add bounding boxes of cls + sep tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    encoding = tokenizer(' '.join(words), return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    bbox = torch.tensor([token_boxes])
    print(
        f"[torch]input_ids shape: f{input_ids.shape}, bbox shape: f{bbox.shape}")

    outputs = model(
        input_ids=input_ids,
        bbox=bbox,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids)
    last_hidden_states = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    return last_hidden_states, pooler_output


if __name__ == "__main__":

    print("\n====test_bert_torch=====")
    torch_hidden_out, torch_pool_out = test_layoutlm_torch()
    torch_hidden_out = torch_hidden_out.detach().numpy()
    torch_pool_out = torch_pool_out.detach().numpy()
    print(torch_hidden_out.shape, torch_pool_out.shape)

    print("\n====test_bert_paddle=====")
    paddle_hidden_out, paddle_pool_out = test_layoutlm_paddle()
    paddle_hidden_out = paddle_hidden_out.numpy()
    paddle_pool_out = paddle_pool_out.numpy()

    # Max absolute difference: 1.9073486e-06
    # Max relative difference: 0.00074042
    np.testing.assert_allclose(torch_hidden_out, paddle_hidden_out)
    # Max absolute difference: 7.1525574e-07
    # Max relative difference: 0.00077766
    np.testing.assert_allclose(torch_pool_out, paddle_pool_out)
