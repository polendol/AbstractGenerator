# Third-party notices

This repository contains training code only. It does not include or
redistribute any model weights, datasets, or third-party source code.
The project relies on the following works, each under its own terms.

## Base model
- Llama 3.2 1B Instruct (Meta Platforms, Inc.), loaded via
  `unsloth/Llama-3.2-1B-Instruct-bnb-4bit`.
  Licensed under the Llama 3.2 Community License Agreement:
  https://www.llama.com/llama3_2/license/
  Subject to the Llama 3.2 Acceptable Use Policy:
  https://www.llama.com/llama3_2/use-policy
  Llama 3.2 is licensed under the Llama 3.2 Community License,
  Copyright © Meta Platforms, Inc. All Rights Reserved.
  Built with Llama.

## Dataset
- arxiv-abstracts-2021 (gfissore): arXiv titles and abstracts up to 2021.
  License: CC0 1.0 Universal (public domain dedication).
  https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021
  Based on: Clement et al., "On the Use of ArXiv as a Dataset", 2019
  (arXiv:1905.00075).

## Libraries
These are installed by the user and are not included in this repository.
Each library's license applies to that library.

 Name         | Version     | License                 | URL                                                     |
|--------------|-------------|-------------------------|---------------------------------------------------------|
| accelerate   | 1.13.0      | Apache Software License | https://github.com/huggingface/accelerate               |
| bitsandbytes | 0.49.2      | MIT                     | https://github.com/bitsandbytes-foundation/bitsandbytes |
| datasets     | 4.3.0       | Apache Software License | https://github.com/huggingface/datasets                 |
| peft         | 0.18.1      | Apache Software License | https://github.com/huggingface/peft                     |
| torch        | 2.6.0+cu124 | BSD License             | https://pytorch.org/                                    |
| transformers | 5.2.0       | Apache 2.0 License      | https://github.com/huggingface/transformers             |
| trl          | 0.24.0      | Apache-2.0              | https://github.com/huggingface/trl                      |
| unsloth      | 2026.3.4    | Apache-2.0              | http://www.unsloth.ai         