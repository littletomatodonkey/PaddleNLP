import numpy as np
import torch
import detectron2

from yacs.config import CfgNode

import paddle


def read_config(fp="cfg.yaml"):
    with open(fp, "r") as fin:
        cfg = CfgNode().load_cfg(fin)
    cfg.freeze()
    return cfg


def detectron2_demo():
    np.random.seed(0)
    img = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255
    # pixel_mean = torch.tensor([])

    img_pt = torch.tensor(img)
    cfg = detectron2.config.get_cfg()
    add_layoutlmv2_config(cfg)

    pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1)
    pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(1, 3, 1, 1)
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg)
    backbone = model.backbone
    backbone.eval()

    pool = torch.nn.AdaptiveAvgPool2d([7, 7])

    img_pt = (img_pt - pixel_mean) / pixel_std

    features = backbone(img_pt)
    print("====backbone output=====")
    for key in features:
        print(key, features[key].shape)

    features = features["p2"]

    pool_res = pool(features)
    print("pool res shape: {}".format(pool_res.shape))

    torch.save(model.state_dict(), "torch_dict.pth")
    # print(backbone)
    return


def paddle_demo():
    np.random.seed(0)
    img = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255
    # pixel_mean = torch.tensor([])

    img_pt = paddle.to_tensor(img)
    cfg = detectron2.config.get_cfg()
    add_layoutlmv2_config(cfg)

    pixel_mean = paddle.to_tensor(cfg.MODEL.PIXEL_MEAN).reshape([1, 3, 1, 1])
    pixel_std = paddle.to_tensor(cfg.MODEL.PIXEL_STD).reshape([1, 3, 1, 1])

    meta_arch = cfg.MODEL.META_ARCHITECTURE

    # just backbone
    # from resnet_paddle import build_resnet_backbone
    # model = build_resnet_backbone(cfg)

    # backbone + fpn
    from fpn_paddle import build_resnet_fpn_backbone
    model = build_resnet_fpn_backbone(cfg)

    model.eval()

    pool = paddle.nn.AdaptiveAvgPool2D(7)

    img_pt = (img_pt - pixel_mean) / pixel_std

    features = model(img_pt)
    print("====model output=====")
    for key in features:
        print(key, features[key].shape)

    features = features["p2"]

    pool_res = pool(features)
    print("pool res shape: {}".format(pool_res.shape))
    # print(model)

    paddle.save(model.state_dict(), "paddle_dict.pdparams")
    return


def trans_detectron2_models():
    torch_layoutlm_weights = "detectron2_layoutlm_torch.pth"
    paddle_layoutlm_weights = "detectron2_layoutlm_paddle.pdparams"

    torch_prefix_name = "layoutlmv2.visual.backbone."

    info_pt = torch.load(torch_layoutlm_weights)

    info_pd = dict()

    for key in info_pt:
        val = info_pt[key].cpu().numpy()
        new_key = key[len(torch_prefix_name):]
        if "num_batches_tracked" in new_key:
            continue
        new_key = new_key.replace("running_mean", "_mean")
        new_key = new_key.replace("running_var", "_variance")
        info_pd[new_key] = val

    print("info pd len: {}".format(len(info_pd)))
    paddle.save(info_pd, paddle_layoutlm_weights)


def test_detectron2_trans():
    """
    test whether the trans is right
    """
    # build input
    np.random.seed(0)
    img = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

    # build cfg
    cfg = read_config("cfg.yaml")

    print(cfg)

    pixel_mean = np.array(
        cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape([1, 3, 1, 1])
    pixel_std = np.array(
        cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape([1, 3, 1, 1])

    img = (img - pixel_mean) / pixel_std

    # build tensor
    img_paddle = paddle.to_tensor(img)
    img_torch = torch.tensor(img).cuda()

    # fetch paddle output
    from fpn_paddle import build_resnet_fpn_backbone
    model_paddle = build_resnet_fpn_backbone(cfg)
    info_pd = paddle.load("detectron2_layoutlm_paddle.pdparams")
    model_paddle.load_dict(info_pd)
    model_paddle.eval()

    paddle_out = model_paddle(img_paddle)

    # fetch torch output
    from fpn_torch import build_resnet_fpn_backbone
    model = build_resnet_fpn_backbone(cfg)
    model_torch = model
    model_torch = model_torch.cuda()

    info_pt = torch.load("detectron2_layoutlm_torch_rm_prefix.pth")
    model_torch.load_state_dict(info_pt)
    model_torch.eval()

    torch_out = model_torch(img_torch)

    print("====backbone output=====")
    for key in paddle_out:
        print("=====test key: {}====".format(key))
        np_paddle_out = paddle_out[key].numpy()
        np_torch_out = torch_out[key].detach().cpu().numpy()
        try:
            # abs diff is very small
            np.testing.assert_allclose(np_paddle_out, np_torch_out)
        except Exception as ex:
            print(ex)
    return


if __name__ == "__main__":
    paddle.set_device("gpu")
    # detectron2_demo()
    # paddle_demo()
    # trans_detectron2_models()
    test_detectron2_trans()
