import torch
import time
import numpy as np
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM


def best_of_n_switch_roles_big_small(
    x, config, prm: PRM,
    big_model, big_tok,
    small_model, small_tok,
    warmup_tokens=200,
    max_prm_context=256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    final_completions = [[] for _ in range(batch_size)]
    final_scores = [[] for _ in range(batch_size)]
    final_tokens = [[] for _ in range(batch_size)]
    final_pred = [None] * batch_size
    tier = [""] * batch_size

    t_big = 0.0
    t_small = 0.0
    t_prm = 0.0

    # ---------------------------
    # Stage 1: Big warm-up
    # ---------------------------
    warm_texts = []
    t0 = time.time()

    for prompt in prompts:
        inp = big_tok(prompt, return_tensors="pt").to(device)
        out = big_model.generate(
            **inp,
            max_new_tokens=warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
        )
        warm_texts.append(big_tok.decode(out[0], skip_special_tokens=True))

    t_big += time.time() - t0

    # Big warm-up PRM baseline
    t1 = time.time()
    warm_scores_raw = prm.score(x["problem"], [[w] for w in warm_texts])
    t_prm += time.time() - t1

    warm_scores = [
        aggregate_scores(s[0], config.agg_strategy) for s in warm_scores_raw
    ]

    # ---------------------------
    # Stage 2: small best-of-n continuation
    # ---------------------------
    small_texts = []
    small_tokens = []

    t2 = time.time()

    for wtxt in warm_texts:
        inp = small_tok(wtxt, return_tensors="pt").to(device)
        out = small_model.generate(
            **inp,
            max_new_tokens=config.max_small_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=config.n,   # 
        )
        decoded = [small_tok.decode(o, skip_special_tokens=True) for o in out]
        small_texts.append(decoded)
        small_tokens.append([o.shape[-1] for o in out])

    t_small += time.time() - t2

    # PRM scoring small best-of-n (sliding window)
    def shorten_for_prm(text):
        ids = small_tok(text, return_tensors="pt").input_ids[0]
        if len(ids) <= max_prm_context:
            return text
        ids = ids[-max_prm_context:]
        return small_tok.decode(ids, skip_special_tokens=True)

    short_small = [
        [shorten_for_prm(t) for t in cand] for cand in small_texts
    ]

    t3 = time.time()
    small_scores_raw = prm.score(x["problem"], short_small)
    t_prm += time.time() - t3

    # Compute best-of-n score for each sample
    small_best_scores = [
        max(aggregate_scores(s, config.agg_strategy) for s in s_list)
        for s_list in small_scores_raw
    ]

    # ---------------------------
    # classify good / bad small
    # ---------------------------
    for i in range(batch_size):
        if small_best_scores[i] >= warm_scores[i]:
            tier[i] = "good_small"
        else:
            tier[i] = "bad_small"

    # ---------------------------
    # Stage 2A: good_small → use small best-of-n
    # ---------------------------
    for i in range(batch_size):
        if tier[i] == "good_small":
            agg_list = [
                aggregate_scores(s, config.agg_strategy)
                for s in small_scores_raw[i]
            ]
            best_idx = int(torch.argmax(torch.tensor(agg_list)))

            final_completions[i] = small_texts[i]
            final_scores[i] = small_scores_raw[i]
            final_tokens[i] = small_tokens[i]
            final_pred[i] = small_texts[i][best_idx]

    # ---------------------------
    # Stage 2B: fallback → big full generation
    # ---------------------------
    t4 = time.time()

    # big-side sliding window for PRM (prevent OOM)
    def shorten_for_prm_big(text):
        ids = big_tok(text, return_tensors="pt").input_ids[0]
        if len(ids) <= max_prm_context:
            return text
        ids = ids[-max_prm_context:]
        return big_tok.decode(ids, skip_special_tokens=True)

    for i in range(batch_size):
        if tier[i] == "bad_small":
            inp = big_tok(warm_texts[i], return_tensors="pt").to(device)
            out = big_model.generate(
                **inp,
                max_new_tokens=config.max_model_len - warmup_tokens,
                do_sample=True,
                top_p=config.top_p,
                temperature=config.temperature,
                num_return_sequences=config.n,
            )

            decoded = [big_tok.decode(o, skip_special_tokens=True) for o in out]
            tokens = [o.shape[-1] for o in out]

            # --- apply sliding window BEFORE PRM scoring ---
            decoded_prm = [shorten_for_prm_big(t) for t in decoded]

            # PRM score on truncated sequences (safe)
            t5 = time.time()
            score_raw = prm.score([x["problem"][i]], [decoded_prm])
            t_prm += time.time() - t5

            agg_list = [
                aggregate_scores(s, config.agg_strategy) for s in score_raw[0]
            ]
            best_idx = int(torch.argmax(torch.tensor(agg_list)))

            final_completions[i] = decoded
            final_scores[i] = score_raw[0]
            final_tokens[i] = tokens
            final_pred[i] = decoded[best_idx]

    t_big += time.time() - t4

    # ---------------------------
    # finalize
    # ---------------------------
    total_time = t_big + t_small + t_prm

    return {
        "completions": final_completions,
        "scores": final_scores,
        "pred": final_pred,
        "completion_tokens": final_tokens,
        "big_gen_time": [t_big] * batch_size,
        "small_gen_time": [t_small] * batch_size,
        "prm_time": [t_prm] * batch_size,
        "total_time": [total_time] * batch_size,
        "tier": tier,
    }
