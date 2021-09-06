import paddle
import numpy as np
import sys
sys.path.insert(0, "../../paddlenlp/transformers/layoutlmv2")

from visual_backbone_config import read_config
from fpn import build_resnet_fpn_backbone


def test_paddle_vb():
    # build input
    np.random.seed(0)
    img = np.random.rand(1, 3, 224, 224).astype(np.float32) * 255

    # build cfg
    cfg = read_config(
        "../../paddlenlp/transformers/layoutlmv2/visual_backbone.yaml")

    pixel_mean = np.array(
        cfg.MODEL.PIXEL_MEAN, dtype=np.float32).reshape([1, 3, 1, 1])
    pixel_std = np.array(
        cfg.MODEL.PIXEL_STD, dtype=np.float32).reshape([1, 3, 1, 1])

    img = (img - pixel_mean) / pixel_std

    # build tensor
    img_paddle = paddle.to_tensor(img)

    # fetch paddle output
    model = build_resnet_fpn_backbone(cfg)
    info = paddle.load("detectron2_layoutlm_paddle.pdparams")
    model.load_dict(info)
    model.eval()

    out = model(img_paddle)
    out = out["p2"]
    print("feature shape: ", out.shape)
    return


if __name__ == "__main__":
    test_paddle_vb()
