# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from paddlenlp.transformers.layoutxlm.tokenization_xlm_roberta import XLMRobertaTokenizer

SPIECE_UNDERLINE = "▁"


class LayoutXLMTokenizer(XLMRobertaTokenizer):
    resource_files_names = {"vocab_file": "sentencepiece.bpe.model"}
    pretrained_resource_files_map = {
        "vocab_file": {
            "layoutxlm-base": "https://huggingface.co/microsoft/layoutxlm-base/resolve/main/sentencepiece.bpe.model",
            "layoutxlm-large": "https://huggingface.co/microsoft/layoutxlm-large/resolve/main/sentencepiece.bpe.model",
        }
    }
    pretrained_init_configuration = {
        "layoutxlm-base": {
            "do_lower_case": False
        },
        "layoutxlm-large": {
            "do_lower_case": False
        },
    }
    pretrained_positional_embedding_sizes = {
        "layoutxlm-base": 512,
        "layoutxlm-large": 512,
    }
    max_model_input_sizes = pretrained_positional_embedding_sizes
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, model_max_length=512, **kwargs):
        super().__init__(model_max_length=model_max_length, **kwargs)


if __name__ == '__main__':
    tokenizer = LayoutXLMTokenizer.from_pretrained('layoutxlm-base')
    text = "i am 中国人"
    result = tokenizer.tokenize(text)
    print(result)
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    result = tokenizer.tokenize(text)
    print(result)