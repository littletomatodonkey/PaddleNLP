# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..layoutlm.tokenizer import LayoutLMTokenizer

__all__ = ['LayoutLMv2Tokenizer', ]


class LayoutLMv2Tokenizer(LayoutLMTokenizer):
    """
    Constructs a BERT tokenizer. It uses a basic tokenizer to do punctuation
    splitting, lower casing and so on, and follows a WordPiece tokenizer to
    tokenize as subwords.
    Args:
        vocab_file (str): file path of the vocabulary
        do_lower_case (bool): Whether the text strips accents and convert to
            lower case. If you use the BERT pretrained model, lower is set to
            Flase when using the cased model, otherwise it is set to True.
            Default: True.
        unk_token (str): The special token for unkown words. Default: "[UNK]".
        sep_token (str): The special token for separator token . Default: "[SEP]".
        pad_token (str): The special token for padding. Default: "[PAD]".
        cls_token (str): The special token for cls. Default: "[CLS]".
        mask_token (str): The special token for mask. Default: "[MASK]".
    
    Examples:
        .. code-block:: python
            from paddle.hapi.text import LayoutLMv2Tokenizer
            tokenizer = LayoutLMv2Tokenizer.from_pretrained('layoutlmv2-base-uncased')
            # the following line get: ['he', 'was', 'a', 'puppet', '##eer']
            tokens = tokenizer('He was a puppeteer')
            # the following line get: 'he was a puppeteer'
            tokenizer.convert_tokens_to_string(tokens)
    """
    resource_files_names = {"vocab_file": "vocab.txt"}  # for save_pretrained
    pretrained_resource_files_map = {
        "vocab_file": {
            "layoutlmv2-base-uncased":
            "https://paddle-hapi.bj.bcebos.com/models/layoutlm/layoutlmv2-base-uncased-vocab.txt",
        }
    }
    pretrained_init_configuration = {
        "layoutlmv2-base-uncased": {
            "do_lower_case": True
        },
    }
    padding_side = 'right'

    pretrained_positional_embeddings_sizes = {"layoutlmv2-base-uncased": 512, }

    def __init__(
            self,
            vocab_file,
            do_lower_case=True,
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
            model_max_length=512, ):
        super().__init__(
            vocab_file,
            do_lower_case=do_lower_case,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token, )
        self.model_max_length = model_max_length
