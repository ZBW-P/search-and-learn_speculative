#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import numpy as np
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from sal.search.best_of_n import best_of_n


#Fail: Vllm kv cache length
def best_of_n_speculative(x, config, llm, prm):
    """
    llm = drafter LLM (small model)
    """

    drafter_output = best_of_n(x, config, llm, prm)

    drafter_completions = drafter_output["completions"]
    drafter_scores = drafter_output["scores"]
    drafter_preds = drafter_output["pred"]
    drafter_tokens = drafter_output["completion_tokens"]

    batch_size = len(x["problem"])

    aggregated_scores = [
        aggregate_scores(score_list, config.agg_strategy)
        for score_list in drafter_scores
    ]

    low_indices = [
        i for i, s in enumerate(aggregated_scores)
        if s < config.spec_threshold
    ]
    if len(low_indices) == 0:
        return drafter_output
    
    low_prompts = [x["problem"][i] for i in low_indices]

    target_llm = config.target_llm

    target_input = {"problem": low_prompts}
    target_output = best_of_n(target_input, config, target_llm, prm)

    target_completions = target_output["completions"]
    target_scores = target_output["scores"]
    target_preds = target_output["pred"]
    target_tokens = target_output["completion_tokens"]


    final_completions = []
    final_scores = []
    final_preds = []
    final_tokens = []

    t_idx = 0 

    for i in range(batch_size):
        if i in low_indices:
            final_completions.append(target_completions[t_idx])
            final_scores.append(target_scores[t_idx])
            final_preds.append(target_preds[t_idx])
            final_tokens.append(target_tokens[t_idx])
            t_idx += 1
        else:
            final_completions.append(drafter_completions[i])
            final_scores.append(drafter_scores[i])
            final_preds.append(drafter_preds[i])
            final_tokens.append(drafter_tokens[i])

    x["completions"] = final_completions
    x["scores"] = final_scores
    x["pred"] = final_preds
    x["completion_tokens"] = final_tokens

    return x
