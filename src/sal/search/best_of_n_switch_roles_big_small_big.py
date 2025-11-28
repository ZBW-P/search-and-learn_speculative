import torch
import time
import numpy as np
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM


def truncate_context(text, tokenizer, max_ctx=2048):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) <= max_ctx:
        return text
    ids = ids[-max_ctx:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def best_of_n_switch_roles_big_small_big(
    x, config, prm: PRM,
    big_model, big_tok,          # 高质量、大模型：只 warm-up + fallback
    small_model, small_tok,      # 小模型：主力生成
    warmup_tokens=128,
    max_prm_context=256,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    t_big = 0.0
    t_small = 0.0
    t_prm = 0.0

    # ============================
    # Stage 1: big model warm-up
    # ============================
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

    # big warm-up 
    t1 = time.time()
    warm_scores_raw = prm.score(x["problem"], [[w] for w in warm_texts])
    t_prm += time.time() - t1
    warm_scores = [
        aggregate_scores(sc[0], config.agg_strategy)
        for sc in warm_scores_raw
    ]

    # ============================
    # Stage 2: small model full generation (best-of-n)
    # ============================
    small_texts = []
    small_tokens = []

    t2 = time.time()

    warm_for_small = [
        truncate_context(w, small_tok, max_ctx=2048) for w in warm_texts
    ]

    small_model.to(device)
    small_model.eval()
    small_model.config.use_cache = False   # ⭐ 关闭 KV cache，防止越跑越慢

    # 1. 批量 tokenize
    enc = small_tok(
        warm_for_small,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        out = small_model.generate(
            **enc,
            max_new_tokens=config.max_small_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=config.n,   # ⭐ 一次性跑 batch * n
            use_cache=False,                 # 再次明确
        )
        # out.shape: [batch_size * n, seq_len]

    # 2. 重新分组
    small_texts = []
    small_tokens = []
    bsz = len(warm_for_small)
    for i in range(bsz):
        cand = out[i * config.n : (i + 1) * config.n]
        decoded = [
            small_tok.decode(o, skip_special_tokens=True)
            for o in cand
        ]
        small_texts.append(decoded)
        small_tokens.append([o.shape[-1] for o in cand])

    t_small += time.time() - t2

    # PRM score 小模型候选（注意截断，防止 PRM OOM）
    def shorten_for_prm(text, tok):
        ids = tok(text, return_tensors="pt").input_ids[0]
        if len(ids) <= max_prm_context:
            return text
        ids = ids[-max_prm_context:]
        return tok.decode(ids, skip_special_tokens=True)

    short_small = [
        [shorten_for_prm(t, small_tok) for t in cand]
        for cand in small_texts
    ]

    t3 = time.time()
    small_scores_raw = prm.score(x["problem"], short_small)
    t_prm += time.time() - t3

    small_agg = [
        [aggregate_scores(s, config.agg_strategy) for s in s_list]
        for s_list in small_scores_raw
    ]
    max_scores = [max(row) for row in small_agg]

    # ============================
    # Stage 2: 计算哪些需要 fallback 到 big
    # ============================
    # 和你原来的逻辑一样：用分位数做阈值
    # T_bad = np.percentile(max_scores, 30)
    # T_abs = np.percentile(max_scores, 10)

    # fallback_indices = [
    #     i for i, sc in enumerate(max_scores)
    #     if sc < T_bad or sc < T_abs
    # ]
    # small_agg: [batch_size][n]
    small_best = [max(row) for row in small_agg]

    fallback_indices = []
    for i in range(batch_size):
        # 如果 small 的最佳评分 < big warmup baseline，就交给 big 兜底
        if small_best[i] < warm_scores[i]:
            fallback_indices.append(i)

    # 如果一个都不用 fallback：全用小模型
    if not fallback_indices:
        pred = [
            small_texts[i][torch.argmax(torch.tensor(small_agg[i]))]
            for i in range(batch_size)
        ]
        total_time = t_big + t_small + t_prm
        return {
            "completions": small_texts,
            "scores": small_scores_raw,
            "pred": pred,
            "completion_tokens": small_tokens,
            "big_gen_time": [t_big] * batch_size,
            "small_gen_time": [t_small] * batch_size,
            "prm_time": [t_prm] * batch_size,
            "total_time": [total_time] * batch_size,
            "tier": ["small_only"] * batch_size,
        }

    # ============================
    # Stage 3: 对低分样本，用 big 重生成（兜底）
    # ============================
    fb_texts = []
    fb_tokens = []
    low_problems = [x["problem"][i] for i in fallback_indices]

    t4 = time.time()
    for idx in fallback_indices:
        w = warm_texts[idx]
        ctx = truncate_context(w, big_tok, max_ctx=4096)
        inp = big_tok(ctx, return_tensors="pt").to(device)
        out = big_model.generate(
            **inp,
            max_new_tokens=config.max_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            num_return_sequences=config.n,
        )
        decoded = [big_tok.decode(o, skip_special_tokens=True) for o in out]
        fb_texts.append(decoded)
        fb_tokens.append([o.shape[-1] for o in out])
    t_big += time.time() - t4

    # big 结果也用 PRM 打分（也用 sliding window）
    fb_texts_prm = [
        [shorten_for_prm(t, big_tok) for t in group]
        for group in fb_texts
    ]

    t5 = time.time()
    fb_scores_raw = prm.score(low_problems, fb_texts_prm)
    t_prm += time.time() - t5

    # ============================
    # 合并 small / big 的结果
    # ============================
    merged_completions = []
    merged_scores = []
    merged_tokens = []
    tier = []

    fb_ptr = 0
    for i in range(batch_size):
        if i in fallback_indices:
            merged_completions.append(fb_texts[fb_ptr])
            merged_scores.append(fb_scores_raw[fb_ptr])
            merged_tokens.append(fb_tokens[fb_ptr])
            tier.append("big_fallback")
            fb_ptr += 1
        else:
            merged_completions.append(small_texts[i])
            merged_scores.append(small_scores_raw[i])
            merged_tokens.append(small_tokens[i])
            tier.append("small_ok")

    final_pred = [
        merged_completions[i][torch.argmax(torch.tensor(
            [aggregate_scores(s, config.agg_strategy)
             for s in merged_scores[i]]
        ))]
        for i in range(batch_size)
    ]

    total_time = t_big + t_small + t_prm

    return {
        "completions": merged_completions,
        "scores": merged_scores,
        "pred": final_pred,
        "completion_tokens": merged_tokens,
        "big_gen_time": [t_big] * batch_size,
        "small_gen_time": [t_small] * batch_size,
        "prm_time": [t_prm] * batch_size,
        "total_time": [total_time] * batch_size,
        "tier": tier,
    }
