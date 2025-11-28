import time
import numpy as np
import torch
from vllm import LLM, SamplingParams

from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM


def truncate_context(text, tokenizer, max_ctx=2048):
    """截断上下文到最后 max_ctx 个 token，防止上下文太长。"""
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if len(ids) <= max_ctx:
        return text
    ids = ids[-max_ctx:]
    return tokenizer.decode(ids, skip_special_tokens=True)


def best_of_n_switch_roles_big_small_big_vllm(
    x, config, prm: PRM,
    big_llm: LLM,              # ⭐ 大模型：vLLM 实例
    small_model, small_tok,    # ⭐ 小模型：HF Transformers
    warmup_tokens: int = 128,
    max_prm_context: int = 256,
):
    """
    Hybrid 版本 big-small-big：
      1) big_llm (vLLM) warm-up
      2) small_model (HF) best-of-n continuation
      3) 对低分样本 fallback 到 big_llm best-of-n
    """
    big_tok = big_llm.get_tokenizer()  # vLLM 里的 HF tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- 构造 prompts --------
    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    t_big = 0.0
    t_small = 0.0
    t_prm = 0.0

    # ============================
    # Stage 1: big model warm-up (vLLM)
    # ============================
    t0 = time.time()

    warmup_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=warmup_tokens,
        n=1,
    )

    warm_resps = big_llm.generate(
        prompts,
        sampling_params=warmup_params,
        use_tqdm=False,
    )

    # vLLM 的 output.text 只包含新生成部分，手动拼回 prompt
    warm_texts = []
    for prompt, resp in zip(prompts, warm_resps):
        gen_text = resp.outputs[0].text  # n=1
        full_text = prompt + gen_text
        warm_texts.append(full_text)

    t_big += time.time() - t0

    # big warm-up PRM baseline
    t1 = time.time()
    warm_scores_raw = prm.score(x["problem"], [[w] for w in warm_texts])
    t_prm += time.time() - t1
    warm_scores = [
        aggregate_scores(sc[0], config.agg_strategy)
        for sc in warm_scores_raw
    ]

    # ============================
    # Stage 2: small model full generation (HF + best-of-n)
    # ============================
    t2 = time.time()

    warm_for_small = [
        truncate_context(w, small_tok, max_ctx=2048) for w in warm_texts
    ]

    small_texts = []
    small_tokens = []

    small_model.to(device)
    small_model.eval()

    with torch.no_grad():
        for ctx in warm_for_small:
            inp = small_tok(ctx, return_tensors="pt").to(device)
            out = small_model.generate(
                **inp,
                max_new_tokens=config.max_small_model_len - warmup_tokens,
                do_sample=True,
                top_p=config.top_p,
                temperature=config.temperature,
                num_return_sequences=config.n,   # ⭐ best-of-n
            )
            # out.shape: [n, seq_len]
            decoded = [
                small_tok.decode(o, skip_special_tokens=True)
                for o in out
            ]
            small_texts.append(decoded)
            small_tokens.append([o.shape[-1] for o in out])

    t_small += time.time() - t2

    # ------- PRM scoring small best-of-n (sliding window) -------
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
    T_bad = np.percentile(max_scores, 30)
    T_abs = np.percentile(max_scores, 10)

    fallback_indices = [
        i for i, sc in enumerate(max_scores)
        if sc < T_bad or sc < T_abs
    ]

    # 如果完全不需要 fallback：全用 small
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
    # Stage 3: 对低分样本，用 big 重生成（兜底，vLLM）
    # ============================
    fb_texts = []
    fb_tokens = []
    low_problems = [x["problem"][i] for i in fallback_indices]

    t4 = time.time()

    fallback_prompts = [warm_texts[idx] for idx in fallback_indices]
    fallback_prompts = [
        truncate_context(p, big_tok, max_ctx=config.max_model_len)
        for p in fallback_prompts
    ]

    fb_params = SamplingParams(
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_model_len - warmup_tokens,
        n=config.n,
    )

    fb_resps = big_llm.generate(
        fallback_prompts,
        sampling_params=fb_params,
        use_tqdm=False,
    )

    for ctx, resp in zip(fallback_prompts, fb_resps):
        cand_texts = []
        cand_tokens = []
        for out in resp.outputs:
            full_txt = ctx + out.text
            cand_texts.append(full_txt)
            cand_tokens.append(len(out.token_ids))
        fb_texts.append(cand_texts)
        fb_tokens.append(cand_tokens)

    t_big += time.time() - t4

    # big 结果也用 PRM 打分（同样 sliding window）
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
