import torch
import time
import numpy as np
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM

# ---------------------------
# 防止 OOM：只保留最后 max_ctx tokens
# ---------------------------
def truncate_context(text, tokenizer, max_ctx=2048):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) <= max_ctx:
        return text
    ids = ids[-max_ctx:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def best_of_n_switch_roles(
    x, config, prm: PRM,
    draft_model, draft_tokenizer,
    target_model, target_tokenizer,
    warmup_tokens=128,
    score_delta=0.05,
    max_prm_context=256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    # ---------------------------
    # prepare output storage
    # ---------------------------
    drafter_completions = [[] for _ in range(batch_size)]
    drafter_tokens = [[] for _ in range(batch_size)]

    t_draft = 0
    t_target = 0
    t_prm = 0

    # ======================================================
    # Stage 1: draft model warm-up
    # ======================================================
    warm_texts = []
    t0 = time.time()

    for prompt in prompts:
        inp = draft_tokenizer(prompt, return_tensors="pt").to(device)
        out = draft_model.generate(
            **inp,
            max_new_tokens=warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=1,
        )
        warm_texts.append(
            draft_tokenizer.decode(out[0], skip_special_tokens=True)
        )

    t_draft += time.time() - t0

    # ======================================================
    # Stage 1 PRM baseline
    # ======================================================
    t1 = time.time()
    warm_scores_raw = prm.score(x["problem"], [[w] for w in warm_texts])
    t_prm += time.time() - t1

    warm_scores = [
        aggregate_scores(sc[0], config.agg_strategy)
        for sc in warm_scores_raw
    ]

    # ======================================================
    # Stage 2: draft model produces full candidate
    # ======================================================
    full_draft_texts = []
    full_draft_tokens = []

    t2 = time.time()
    for w in warm_texts:
        ctx = truncate_context(w, draft_tokenizer, max_ctx=2048)
        inp = draft_tokenizer(ctx, return_tensors="pt").to(device)
        out = draft_model.generate(
            **inp,
            max_new_tokens=config.max_small_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=config.n,
        )
        decoded = [
            draft_tokenizer.decode(o, skip_special_tokens=True)
            for o in out
        ]
        full_draft_texts.append(decoded)
        full_draft_tokens.append([o.shape[-1] for o in out])

    t_draft += time.time() - t2

    # ======================================================
    # Stage 2 PRM score draft
    # ======================================================
    # PRM 可能 OOM——所以 sliding window
    def shorten_for_prm(text):
        ids = draft_tokenizer(text, return_tensors="pt").input_ids[0]
        if len(ids) <= max_prm_context:
            return text
        ids = ids[-max_prm_context:]
        return draft_tokenizer.decode(ids, skip_special_tokens=True)

    short_draft = [[shorten_for_prm(t) for t in cand]
                   for cand in full_draft_texts]

    t3 = time.time()
    draft_scores_raw = prm.score(x["problem"], short_draft)
    t_prm += time.time() - t3

    agg_scores = [
        [aggregate_scores(s, config.agg_strategy) for s in s_list]
        for s_list in draft_scores_raw
    ]
    max_scores = [max(row) for row in agg_scores]

    # ======================================================
    # Stage 2: 判断 fallback
    # ======================================================
    T_bad = np.percentile(max_scores, 30)       # your logic
    T_abs = np.percentile(max_scores, 10)

    fallback_indices = [
        i for i, sc in enumerate(max_scores)
        if sc < T_bad or sc < T_abs
    ]

    if not fallback_indices:
        # pure fast path
        pred = [
            full_draft_texts[i][torch.argmax(torch.tensor(agg_scores[i]))]
            for i in range(batch_size)
        ]
        return {
            "completions": full_draft_texts,
            "scores": draft_scores_raw,
            "pred": pred,
            "completion_tokens": full_draft_tokens,
            "draft_gen_time": [t_draft] * batch_size,
            "prm_score_time_draft": [t_prm] * batch_size,
            "target_gen_time": [0.0] * batch_size,
            "prm_score_time_target": [0.0] * batch_size,
            "total_time": [t_draft + t_prm] * batch_size,
        }

    # ======================================================
    # Stage 3: fallback only for low-quality samples
    # ======================================================
    fb_texts = []
    fb_tokens = []
    low_problems = [x["problem"][i] for i in fallback_indices]

    t4 = time.time()
    for idx in fallback_indices:
        w = warm_texts[idx]
        ctx = truncate_context(w, target_tokenizer, max_ctx=4096)
        inp = target_tokenizer(ctx, return_tensors="pt").to(device)
        out = target_model.generate(
            **inp,
            max_new_tokens=config.max_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=config.n,
        )
        decoded = [
            target_tokenizer.decode(o, skip_special_tokens=True)
            for o in out
        ]
        fb_texts.append(decoded)
        fb_tokens.append([o.shape[-1] for o in out])

    t_target += time.time() - t4

    # sliding window PRM
    def shorten_fb(texts):
        r = []
        for t in texts:
            ids = target_tokenizer(t, return_tensors="pt").input_ids[0]
            if len(ids) <= max_prm_context:
                r.append(t)
            else:
                ids = ids[-max_prm_context:]
                r.append(target_tokenizer.decode(ids, skip_special_tokens=True))
        return r

    fb_texts_prm = [shorten_fb(group) for group in fb_texts]

    t5 = time.time()
    fb_scores_raw = prm.score(low_problems, fb_texts_prm)
    t_prm += time.time() - t5

    # ======================================================
    # Merge everything
    # ======================================================
    merged_completions = []
    merged_scores = []
    merged_tokens = []

    fb_ptr = 0
    for i in range(batch_size):
        if i in fallback_indices:
            merged_completions.append(fb_texts[fb_ptr])
            merged_scores.append(fb_scores_raw[fb_ptr])
            merged_tokens.append(fb_tokens[fb_ptr])
            fb_ptr += 1
        else:
            merged_completions.append(full_draft_texts[i])
            merged_scores.append(draft_scores_raw[i])
            merged_tokens.append(full_draft_tokens[i])

    final_pred = [
        merged_completions[i][torch.argmax(torch.tensor(
            [aggregate_scores(s, config.agg_strategy)
             for s in merged_scores[i]]
        ))]
        for i in range(batch_size)
    ]

    total_time = t_draft + t_target + t_prm

    return {
        "completions": merged_completions,
        "scores": merged_scores,
        "pred": final_pred,
        "completion_tokens": merged_tokens,
        "draft_gen_time": [t_draft] * batch_size,
        "prm_score_time_draft": [t_prm] * batch_size,
        "target_gen_time": [t_target] * batch_size,
        "prm_score_time_target": [0.0] * batch_size,
        "total_time": [total_time] * batch_size,
    }
