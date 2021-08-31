import os
from yacs.config import CfgNode


def read_config(fp=None):
    if fp is None:
        dir_name = os.path.dirname(os.path.abspath(__file__))
        fp = os.path.join(dir_name, "visual_backbone.yaml")
    with open(fp, "r") as fin:
        cfg = CfgNode().load_cfg(fin)
    cfg.freeze()
    return cfg
