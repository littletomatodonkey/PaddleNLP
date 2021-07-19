import paddle
import torch
import numpy as np

# original bert pretrained
# optional
ori_weights_paddle = paddle.load(
    "/root/.paddlenlp/models/bert-base-uncased/bert-base-uncased.pdparams.ori")
# torch weights
weights_torch = torch.load("bert-base-uncased-torch/pytorch_model.bin")

save_paddle_path = "./bert-base-uncased.pdparams"


def get_paddle_encoder_param_name(torch_name):
    """
    name starts with "bert.encoder.layer"
    """
    torch_to_paddle_dict = {
        # encoder
        "attention.self.query.weight": "self_attn.q_proj.weight",
        "attention.self.query.bias": "self_attn.q_proj.bias",
        "attention.self.key.weight": "self_attn.k_proj.weight",
        "attention.self.key.bias": "self_attn.k_proj.bias",
        "attention.self.value.weight": "self_attn.v_proj.weight",
        "attention.self.value.bias": "self_attn.v_proj.bias",
        "attention.output.dense.weight": "self_attn.out_proj.weight",
        "attention.output.dense.bias": "self_attn.out_proj.bias",
        "attention.output.LayerNorm.gamma": "norm1.weight",
        "attention.output.LayerNorm.beta": "norm1.bias",
        "intermediate.dense.weight": "linear1.weight",
        "intermediate.dense.bias": "linear1.bias",
        "output.dense.weight": "linear2.weight",
        "output.dense.bias": "linear2.bias",
        "output.LayerNorm.gamma": "norm2.weight",
        "output.LayerNorm.beta": "norm2.bias",
        # cls pred
        "cls.predictions.transform.dense.weight":
        "cls.predictions.transform.weight",
        "cls.predictions.transform.dense.bias":
        "cls.predictions.transform.bias",
        "cls.predictions.transform.LayerNorm.gamma":
        "cls.predictions.layer_norm.weight",
        "cls.predictions.transform.LayerNorm.beta":
        "cls.predictions.layer_norm.bias",
        "cls.predictions.decoder.weight": "cls.predictions.decoder_weight",
        "cls.predictions.bias": "cls.predictions.decoder_bias",
    }
    # paddle: bert.encoder.layers.2.self_attn.q_proj.weight
    # torch : bert.encoder.layer.2.attention.self.query.weight

    encoder_flag = "bert.encoder.layer"
    embedding_flag = "bert.embeddings"
    cls_pred_flag = "cls.predictions"
    if torch_name.startswith(encoder_flag):
        paddle_name = torch_name[:len(encoder_flag)] + "s."
        torch_name = torch_name[len(encoder_flag) + 1:]

        number = torch_name.split(".")[0]
        paddle_name += number + "."

        torch_name = torch_name[len(number) + 1:]
        paddle_name += torch_to_paddle_dict[torch_name]
    elif torch_name.startswith(embedding_flag):
        paddle_name = torch_name.replace("LayerNorm.gamma", "layer_norm.weight")
        paddle_name = paddle_name.replace("LayerNorm.beta", "layer_norm.bias")
    elif torch_name.startswith(cls_pred_flag):
        paddle_name = torch_to_paddle_dict[torch_name]
    else:
        paddle_name = torch_name

    return paddle_name


new_weight_paddle = dict()
for key_torch in weights_torch.keys():
    val = weights_torch[key_torch].detach().numpy()
    if len(val.shape) == 2 and not "embedding" in key_torch.lower():
        val = np.transpose(val)

    key_paddle = get_paddle_encoder_param_name(key_torch)
    print(key_torch, key_paddle)
    assert key_paddle in ori_weights_paddle

    new_weight_paddle[key_paddle] = val

paddle.save(new_weight_paddle, save_paddle_path)
