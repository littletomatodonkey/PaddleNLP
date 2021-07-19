import sys
import os
import paddle
import torch
import numpy as np

sys.path.insert(0, "./layoutlm_paddle/PaddleNLP")
# sys.path.insert(0, "./layoutlm_torch")

import transformers


def get_input_demo():
    words = ["train data", "eval data", "train and eval data"]
    # words = ["train and eval data"]
    normalized_word_boxes = [637, 773, 693, 782], [698, 773, 733, 782]
    # input_ids = np.array([[ 101, 3345, 2951,  102,    0,    0,    0],
    #     [ 101, 9345, 2140, 2951,  102,    0,    0],
    #     [ 101, 3345, 1998, 9345, 2140, 2951,  102]])
    # input_ids = [[ 101, ]]
    return words, normalized_word_boxes


def preprcess_data_paddle(x_list):
    from paddlenlp.data import Tuple, Pad
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


def test_bert_paddle():
    from paddlenlp.transformers import BertTokenizer, BertModel

    # tokenizer = BertTokenizer.from_pretrained("./bert-base-uncased-paddle/")
    # model = BertModel.from_pretrained("./bert-base-uncased-paddle/")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased-paddle")
    model = BertModel.from_pretrained("bert-base-uncased-paddle")
    model.eval()
    words, _ = get_input_demo()

    # tokenizer.save_pretrained("test2")
    # model.save_pretrained("test2")

    tokenizer_res = tokenizer(
        words,
        pad_to_max_seq_len=False,
        truncation_strategy="longest_first",
        max_seq_len=10,
        return_attention_mask=True)

    tokenizer_res = preprcess_data_paddle(tokenizer_res)
    print(tokenizer_res)

    output = model(paddle.to_tensor(tokenizer_res["input_ids"]))

    # output = model(paddle.to_tensor(paddle.to_tensor(input_ids)))

    # last_hidden_state
    # print(output[0].shape) # batch_size x 7 x 768
    # pooler_output batch_size x 768
    # print(output[1].shape)

    return output[0], output[1]


def test_bert_torch():
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")
    model.eval()
    # input_ids, normalized_word_boxes = get_input_demo()
    words, normalized_word_boxes = get_input_demo()

    tokenizer_res = tokenizer(
        words,
        padding=True,
        truncation=True,
        max_length=10,
        return_tensors="pt")
    print(tokenizer_res)
    output = model(tokenizer_res["input_ids"], tokenizer_res["attention_mask"])

    # last_hidden_state
    # print(output[0].shape) # batch_size x 7 x 768
    # pooler_output batch_size x 768
    # print(output[1].shape)

    return output[0], output[1]


if __name__ == "__main__":
    print("\n====test_bert_torch=====")
    torch_hidden_out, torch_pool_out = test_bert_torch()
    torch_hidden_out = torch_hidden_out.detach().numpy()
    torch_pool_out = torch_pool_out.detach().numpy()

    print("\n====test_bert_paddle=====")
    paddle_hidden_out, paddle_pool_out = test_bert_paddle()
    paddle_hidden_out = paddle_hidden_out.numpy()
    paddle_pool_out = paddle_pool_out.numpy()

    # ut will not pass because
    # Max absolute difference: 3.8146973e-06
    # Max relative difference: 0.00388905
    np.testing.assert_allclose(torch_hidden_out, paddle_hidden_out)
    # Max absolute difference: 8.6426735e-07
    # Max relative difference: 2.0692234e-05
    np.testing.assert_allclose(torch_pool_out, paddle_pool_out)
