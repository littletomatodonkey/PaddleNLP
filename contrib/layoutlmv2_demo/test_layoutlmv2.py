import sys
import os
import paddle
import torch
import numpy as np

sys.path.insert(0, "../../")

torch_model_dir = "microsoft/layoutlm-base-uncased/"


def get_input_demo(platform="paddle", device="cpu"):
    info = paddle.load("fake_input_paddle.data")
    # imgs = np.random.rand(info["input_ids"].shape[0], 3, 224, 224).astype(np.float32)
    # info["image"] = paddle.to_tensor(imgs)
    if platform == "torch":
        info = {key: torch.tensor(info[key].numpy()) for key in info}
        if device == "gpu":
            info = {key: info[key].cuda() for key in info}
    return info


def test_layoutlm_paddle():
    from paddlenlp.transformers import LayoutLMv2Model, LayoutLMv2Tokenizer

    tokenizer = LayoutLMv2Tokenizer.from_pretrained(
        "./layoutlmv2-base-uncased-paddle/")
    model = LayoutLMv2Model.from_pretrained("./layoutlmv2-base-uncased-paddle/")
    model.eval()

    paddle.save(model.state_dict(), "v2.pdparams")

    batch_input = get_input_demo(platform="paddle", device="gpu")

    outputs = model(
        input_ids=batch_input["input_ids"],
        bbox=batch_input["bbox"],
        image=batch_input["image"],
        attention_mask=batch_input["attention_mask"], )
    sequence_output = outputs[0]
    pooled_output = outputs[1]
    return sequence_output, pooled_output


def test_layoutlm_torch():
    # import pytorch models
    from layoutlmft.models.layoutlmv2 import LayoutLMv2Model, LayoutLMv2Tokenizer

    model_dir = "./layoutlmv2-base-uncased-torch/"

    model = LayoutLMv2Model.from_pretrained(model_dir)
    model.eval()
    model = model.cuda()

    batch_input = get_input_demo(platform="torch", device="gpu")

    outputs = model(
        input_ids=batch_input["input_ids"],
        bbox=batch_input["bbox"],
        image=batch_input["image"],
        attention_mask=batch_input["attention_mask"], )
    sequence_output = outputs[0]
    pooled_output = outputs[1]
    return sequence_output, pooled_output


if __name__ == "__main__":

    #print("\n====test_layoutlmv2_torch=====")
    #torch_hidden_out, torch_pool_out = test_layoutlm_torch()
    #torch_hidden_out = torch_hidden_out.cpu().detach().numpy()
    #torch_pool_out = torch_pool_out.cpu().detach().numpy()
    #print(torch_hidden_out.shape, torch_pool_out.shape)

    print("\n====test_layoutlmv2_paddle=====")
    paddle_hidden_out, paddle_pool_out = test_layoutlm_paddle()
    paddle_hidden_out = paddle_hidden_out.numpy()
    paddle_pool_out = paddle_pool_out.numpy()
    print(paddle_hidden_out.shape, paddle_pool_out.shape)

    # Max absolute difference: 0.002334
    # Max relative difference: 7.8151183
    #np.testing.assert_allclose(torch_hidden_out, paddle_hidden_out)
    # Max absolute difference: 1.66893e-06
    # Max relative difference: 0.02247073
    # np.testing.assert_allclose(torch_pool_out, paddle_pool_out)
