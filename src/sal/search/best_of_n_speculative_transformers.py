import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM
import numpy as np
import time

def best_of_n_speculative_transformers(
    x, config, prm: PRM,
    draft_model, draft_tokenizer,
    target_model, target_tokenizer
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        f"{config.system_prompt}\n{p}"
        for p in x["problem"]
    ]

    batch_size = len(prompts)

    drafter_completions = [[] for _ in range(batch_size)]
    drafter_tokens = [[] for _ in range(batch_size)]

    # ----------------- Draft model generation -----------------
    t_draft_start = time.time()
    for i, prompt in enumerate(prompts):
        inputs = draft_tokenizer(prompt, return_tensors="pt").to(device)

        outputs = draft_model.generate(
            **inputs,
            max_new_tokens=config.max_small_model_len,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            num_return_sequences=config.n,
        )

        decoded = [
            draft_tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]
        drafter_completions[i] = decoded
        drafter_tokens[i] = [out.shape[-1] for out in outputs]

    t_draft_end = time.time()
    draft_gen_time = t_draft_end - t_draft_start

    # ----------------- PRM scoring for draft -----------------
    t_prm1_start = time.time()
    draft_scores = prm.score(x["problem"], drafter_completions)
    t_prm1_end = time.time()
    prm_score_time_draft = t_prm1_end - t_prm1_start

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score_list]
        for score_list in draft_scores
    ]
    max_scores = [max(s) for s in agg_scores]

    T_low = np.percentile(max_scores, 30)
    T_absolute = np.percentile(max_scores, 10)

    low_indices = [
        i for i, sc in enumerate(max_scores)
        if sc < T_low or sc < T_absolute
    ]

    # ------------------ If no fallback needed ------------------
    if len(low_indices) == 0:
        pred = [
            drafter_completions[i][torch.argmax(torch.tensor(agg_scores[i]))]
            for i in range(batch_size)
        ]

        return {
            "completions": drafter_completions,
            "scores": draft_scores,
            "pred": pred,
            "completion_tokens": drafter_tokens,

            # batched=True requires lists
            "draft_gen_time": [draft_gen_time] * batch_size,
            "prm_score_time_draft": [prm_score_time_draft] * batch_size,
            "target_gen_time": [0.0] * batch_size,
            "prm_score_time_target": [0.0] * batch_size,
            "total_time": [(draft_gen_time + prm_score_time_draft)] * batch_size,
        }

    # ------------------ Fallback with target model ------------------
    t_target_start = time.time()
    fallback_completions = []
    fallback_tokens = []

    for idx in low_indices:
        inputs = target_tokenizer(prompts[idx], return_tensors="pt").to(device)

        outputs = target_model.generate(
            **inputs,
            max_new_tokens=config.max_model_len,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            num_return_sequences=config.n,
        )

        decoded = [
            target_tokenizer.decode(out, skip_special_tokens=True)
            for out in outputs
        ]

        fallback_completions.append(decoded)
        fallback_tokens.append([out.shape[-1] for out in outputs])

    t_target_end = time.time()
    target_gen_time = t_target_end - t_target_start

    # ------------------ PRM scoring for fallback ------------------
    low_problems = [x["problem"][i] for i in low_indices]

    t_prm2_start = time.time()
    fallback_scores_raw = prm.score(low_problems, fallback_completions)
    t_prm2_end = time.time()
    prm_score_time_target = t_prm2_end - t_prm2_start

    # ------------------ Merge draft/fallback ------------------
    final_completions = []
    final_scores = []
    final_tokens = []

    fb_ptr = 0
    for i in range(batch_size):
        if i in low_indices:
            final_completions.append(fallback_completions[fb_ptr])
            final_scores.append(fallback_scores_raw[fb_ptr])
            final_tokens.append(fallback_tokens[fb_ptr])
            fb_ptr += 1
        else:
            final_completions.append(drafter_completions[i])
            final_scores.append(draft_scores[i])
            final_tokens.append(drafter_tokens[i])

    final_pred = [
        final_completions[i][torch.argmax(torch.tensor([
            aggregate_scores(s, config.agg_strategy)
            for s in final_scores[i]
        ]))]
        for i in range(batch_size)
    ]

    total_time = draft_gen_time + prm_score_time_draft + target_gen_time + prm_score_time_target

    # ------------------ Return batched output ------------------
    return {
        "completions": final_completions,
        "scores": final_scores,
        "pred": final_pred,
        "completion_tokens": final_tokens,

        "draft_gen_time": [draft_gen_time] * batch_size,
        "prm_score_time_draft": [prm_score_time_draft] * batch_size,
        "target_gen_time": [target_gen_time] * batch_size,
        "prm_score_time_target": [prm_score_time_target] * batch_size,
        "total_time": [total_time] * batch_size,
    }
