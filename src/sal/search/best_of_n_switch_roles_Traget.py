import torch
import torch.nn.functional as F
import time
import numpy as np
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM


def compute_logprob(model, tokenizer, prompt_text, candidate_text):
    """
    返回 logprob(candidate_text | prompt_text)
    """
    device = next(model.parameters()).device

    full = prompt_text + candidate_text
    tokens = tokenizer(full, return_tensors="pt").to(device)

    input_ids = tokens["input_ids"]
    target_ids = input_ids.clone()

    # mask 掉 prompt 部分，只计算 candidate 的概率
    prompt_len = len(tokenizer(prompt_text)["input_ids"])
    target_ids[:, :prompt_len] = -100  # ignore_index

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        # out.loss 是 masked 的 cross entropy
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_tokens = target_ids[:, 1:]

    mask = target_tokens != -100
    selected = log_probs[0].gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
    neg_log_likelihood = -selected[mask].mean().item()

    # 越大越好，所以取 logprob = - loss
    return -neg_log_likelihood



def best_of_n_switch_roles_Traget(
    x, config, prm: PRM,
    big_model, big_tok,
    small_model, small_tok,
    warmup_tokens=200,
    score_delta=0.05,
    high_margin=0.0,
    chunk_size=32,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    final_completions = [[] for _ in range(batch_size)]
    final_scores = [[] for _ in range(batch_size)]
    final_tokens = [[] for _ in range(batch_size)]
    final_pred = [None] * batch_size

    t_big_gen = 0.0
    t_small_gen = 0.0
    t_prm = 0.0

    # ---------------------------
    # Stage 1: big_model warm-up
    # ---------------------------
    warmup_texts = []
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
        warmup_texts.append(big_tok.decode(out[0], skip_special_tokens=True))

    t_big_gen += time.time() - t0

    # baseline score
    t1 = time.time()
    big_scores_raw = prm.score(x["problem"], [[w] for w in warmup_texts])
    t_prm += (time.time() - t1)

    big_baseline = [
        aggregate_scores(s[0], config.agg_strategy)
        for s in big_scores_raw
    ]

    # ---------------------------
    # Stage 2: small full generation
    # ---------------------------
    small_full = []
    t2 = time.time()

    for wtxt in warmup_texts:
        inp = small_tok(wtxt, return_tensors="pt").to(device)
        out = small_model.generate(
            **inp,
            max_new_tokens=config.max_small_model_len - warmup_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
        )
        small_full.append(small_tok.decode(out[0], skip_special_tokens=True))

    t_small_gen += time.time() - t2

    # PRM scoring small full
    t3 = time.time()
    small_scores_raw = prm.score(x["problem"], [[w] for w in small_full])
    t_prm += time.time() - t3

    small_scores = [
        aggregate_scores(s[0], config.agg_strategy)
        for s in small_scores_raw
    ]

    # 分类 tier
    tier = []
    for i in range(batch_size):
        if small_scores[i] >= big_baseline[i] + high_margin:
            tier.append("good_small")
        elif small_scores[i] >= big_baseline[i] - score_delta:
            tier.append("mid_small")
        else:
            tier.append("bad_small")

    # ---------------------------
    # Stage 3A: good_small
    # ---------------------------
    for i in range(batch_size):
        if tier[i] == "good_small":
            final_completions[i] = [small_full[i]]
            final_scores[i] = [small_scores_raw[i][0]]
            final_tokens[i] = [len(small_full[i].split())]
            final_pred[i] = small_full[i]

    # ---------------------------
    # Stage 3B: bad_small → big model regenerate
    # ---------------------------
    t4 = time.time()
    for i in range(batch_size):
        if tier[i] == "bad_small":
            inp = big_tok(warmup_texts[i], return_tensors="pt").to(device)
            out = big_model.generate(
                **inp,
                max_new_tokens=config.max_model_len - warmup_tokens,
                do_sample=True,
                top_p=config.top_p,
                temperature=config.temperature,
            )
            full_txt = big_tok.decode(out[0], skip_special_tokens=True)

            # score full
            t5 = time.time()
            score_raw = prm.score([x["problem"][i]], [[full_txt]])
            t_prm += time.time() - t5

            final_completions[i] = [full_txt]
            final_scores[i] = [score_raw[0][0]]
            final_tokens[i] = [len(full_txt.split())]
            final_pred[i] = full_txt

    t_big_gen += time.time() - t4

    # ---------------------------
    # Stage 3C: mid_small → chunk verify using target-model logprob
    # ---------------------------
    for i in range(batch_size):
        if tier[i] != "mid_small":
            continue

        cur = warmup_texts[i]

        max_new = config.max_model_len - warmup_tokens
        max_chunks = max_new // chunk_size

        for k in range(max_chunks):
            # small chunk
            inp_small = small_tok(cur, return_tensors="pt").to(device)
            out_small = small_model.generate(
                **inp_small,
                max_new_tokens=chunk_size,
                do_sample=True,
                top_p=config.top_p,
                temperature=config.temperature,
            )
            chunk_small = small_tok.decode(out_small[0], skip_special_tokens=True)

            # -------------------
            # target logprob verify (关键替代 PRM)
            # -------------------
            lp_small = compute_logprob(
                big_model, big_tok,
                cur,
                chunk_small
            )

            # heuristic threshold
            if lp_small > -3.0:  # 越大越好，可调
                cur += chunk_small
            else:
                # fallback big
                inp_big = big_tok(cur, return_tensors="pt").to(device)
                out_big = big_model.generate(
                    **inp_big,
                    max_new_tokens=chunk_size,
                    do_sample=True,
                    top_p=config.top_p,
                    temperature=config.temperature,
                )
                chunk_big = big_tok.decode(out_big[0], skip_special_tokens=True)
                cur += chunk_big

        final_pred[i] = cur
        final_completions[i] = [cur]
        final_scores[i] = [[]]
        final_tokens[i] = [len(cur.split())]

    total_time = t_big_gen + t_small_gen + t_prm

    return {
        "completions": final_completions,
        "scores": final_scores,
        "pred": final_pred,
        "completion_tokens": final_tokens,
        "big_gen_time": [t_big_gen] * batch_size,
        "small_gen_time": [t_small_gen] * batch_size,
        "prm_time": [t_prm] * batch_size,
        "total_time": [total_time] * batch_size,
        "tier": tier,
    }
