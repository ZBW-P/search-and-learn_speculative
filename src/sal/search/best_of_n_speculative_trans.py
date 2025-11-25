import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM

def best_of_n_speculative_transformers(
    x, config, prm: PRM,
    draft_model, draft_tokenizer,
    target_model, target_tokenizer
):
    """
    Speculative decoding using Transformers:
    1) drafter_model generates N candidates
    2) PRM scores them
    3) low-score prompts fall back to target_model for regeneration
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [
        f"{config.system_prompt}\n{p}"
        for p in x["problem"]
    ]

    batch_size = len(prompts)

    drafter_completions = [[] for _ in range(batch_size)]
    drafter_tokens = [[] for _ in range(batch_size)]

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

    draft_scores = prm.score(x["problem"], drafter_completions)

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in score_list]
        for score_list in draft_scores
    ]

    low_indices = [
        i for i, score_list in enumerate(agg_scores)
        if max(score_list) < config.spec_threshold
    ]

    if len(low_indices) == 0:
        # best-of-n prediction
        pred = [
            drafter_completions[i][torch.argmax(torch.tensor(agg_scores[i]))]
            for i in range(batch_size)
        ]
        return {
            "completions": drafter_completions,
            "scores": draft_scores,
            "pred": pred,
            "completion_tokens": drafter_tokens
        }

    fallback_completions = []
    fallback_tokens = []
    fallback_scores = []

    for idx in low_indices:
        prompt = prompts[idx]
        inputs = target_tokenizer(prompt, return_tensors="pt").to(device)

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

    low_problems = [x["problem"][i] for i in low_indices]
    fallback_scores = prm.score(low_problems, fallback_completions)

    final_completions = []
    final_scores = []
    final_tokens = []

    fb_ptr = 0
    for i in range(batch_size):
        if i in low_indices:
            final_completions.append(fallback_completions[fb_ptr])
            final_scores.append(fallback_scores[fb_ptr])
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

    return {
        "completions": final_completions,
        "scores": final_scores,
        "pred": final_pred,
        "completion_tokens": final_tokens,
    }
