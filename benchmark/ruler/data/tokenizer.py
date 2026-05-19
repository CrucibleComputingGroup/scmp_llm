# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from transformers import AutoTokenizer


def select_tokenizer(tokenizer_type: str, tokenizer_path: str) -> "HFTokenizer":
    if tokenizer_type == "hf":
        return HFTokenizer(model_path=tokenizer_path)
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


class HFTokenizer:
    """
    Tokenizer from HF models
    """

    def __init__(self, model_path: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def text_to_tokens(self, text: str) -> list[str]:
        tokens: list[str] = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: list[int]) -> str:
        text: str = self.tokenizer.convert_tokens_to_string(tokens)
        return text
