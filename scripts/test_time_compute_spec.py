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

import logging
import os
import pickle
import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.search.best_of_n_speculative_transformers import best_of_n_speculative_transformers
from sal.search.best_of_n_switch_roles_small_big import best_of_n_switch_roles_small_big
from sal.search.best_of_n_switch_roles_big_small import best_of_n_switch_roles_big_small
from sal.search.best_of_n_switch_roles_big_small_big import best_of_n_switch_roles_big_small_big
from sal.search.best_of_n_switch_roles_big_small_big_vllm import (
    best_of_n_switch_roles_big_small_big_vllm,
)
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score
from datetime import datetime

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
    "best_of_n_speculative_transformers": best_of_n_speculative_transformers,
    "best_of_n_switch_roles_small_big": best_of_n_switch_roles_small_big, 
    "best_of_n_switch_roles_big_small":best_of_n_switch_roles_big_small,
    "best_of_n_switch_roles_big_small_big": best_of_n_switch_roles_big_small_big,
    "best_of_n_switch_roles_big_small_big_vllm": best_of_n_switch_roles_big_small_big_vllm,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()

    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    
    prm = load_prm(config)
       
    if config.approach == "best_of_n_speculative_transformers":
        
        if config.small_model_path is None:
            raise ValueError(
                "best_of_n_speculative_transformers requires --small_model_path to be set."
            )
        
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load target model
        target_tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        target_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Load draft model
        draft_tokenizer = AutoTokenizer.from_pretrained(config.small_model_path)
        draft_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        fn_kwargs = {
            "config": config,
            "prm": prm,
            "target_model": target_model,
            "target_tokenizer": target_tokenizer,
            "draft_model": draft_model,
            "draft_tokenizer": draft_tokenizer,
        }
    elif config.approach == "best_of_n_switch_roles_small_big":

        if config.small_model_path is None:
            raise ValueError(
                "best_of_n_switch_roles_small_big requires --small_model_path to be set."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # --- Load BIG model (high-quality warm-up generator) ---
        big_tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        big_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # --- Load SMALL model (fast generator / fallback) ---
        small_tokenizer = AutoTokenizer.from_pretrained(config.small_model_path)
        small_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        fn_kwargs = {
            "config": config,
            "prm": prm,
            "draft_model": small_model,
            "draft_tokenizer": small_tokenizer,
            "target_model": big_model,
            "target_tokenizer": big_tokenizer,
        }
    elif config.approach == "best_of_n_switch_roles_big_small":

        if config.small_model_path is None:
            raise ValueError(
                "best_of_n_switch_roles_big_small requires --small_model_path to be set."
            )

        from transformers import AutoModelForCausalLM, AutoTokenizer

        # --- Load BIG model (high-quality warm-up generator) ---
        big_tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        big_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # --- Load SMALL model (fast generator / fallback) ---
        small_tokenizer = AutoTokenizer.from_pretrained(config.small_model_path)
        small_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        fn_kwargs = {
            "config": config,
            "prm": prm,
            "big_model": big_model,
            "big_tok": big_tokenizer,
            "small_model": small_model,
            "small_tok": small_tokenizer,
        }
    elif config.approach == "best_of_n_switch_roles_big_small_big":
        from transformers import AutoModelForCausalLM, AutoTokenizer
    
        big_tok = AutoTokenizer.from_pretrained(config.model_path)
        big_model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
        small_tok = AutoTokenizer.from_pretrained(config.small_model_path)
        small_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
        fn_kwargs = {
            "config": config,
            "prm": prm,
            "big_model": big_model,
            "big_tok": big_tok,
            "small_model": small_model,
            "small_tok": small_tok,
        }
    elif config.approach == "best_of_n_switch_roles_big_small_big_vllm":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vllm import LLM
    
        big_llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            dtype="bfloat16",
            seed=config.seed,
        )
    
        small_tok = AutoTokenizer.from_pretrained(config.small_model_path)
        small_model = AutoModelForCausalLM.from_pretrained(
            config.small_model_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
        fn_kwargs = {
            "config": config,
            "prm": prm,
            "big_llm": big_llm,
            "small_model": small_model,
            "small_tok": small_tok,
        }
    else:
        from vllm import LLM
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=config.seed,
            max_model_len=config.max_model_len,
            tensor_parallel_size=num_gpus,
        )

        fn_kwargs = {
            "config": config,
            "llm": llm,
            "prm": prm,
        }

    dataset = get_dataset(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs=fn_kwargs,
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score(dataset, config)
    pkl_folder = "pkl_results"
    os.makedirs(pkl_folder, exist_ok=True)
    
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = config.model_path.split('/')[-1]
    approach_name = config.approach
    pickle_filename = os.path.join(pkl_folder, f"timing_{model_name}_{approach_name}_{time_str}.pkl")
    
    try:
        with open(pickle_filename, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved all timing results to {pickle_filename}")
    except Exception as e:
        logger.error(f"Failed to save timing results: {e}")

    save_dataset(dataset, config)
    logger.info("Done 🔥!")


if __name__ == "__main__":
    main()
