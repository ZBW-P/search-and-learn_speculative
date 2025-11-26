import torch
import time
import numpy as np
from sal.utils.score import aggregate_scores
from sal.models.reward_models import PRM

def best_of_n_switch_roles(
    x, config, prm: PRM,
    big_model, big_tok,
    small_model, small_tok,
    warmup_tokens=200,      # Stage 1: 大模型起手长度
    score_delta=0.05,       # 中档阈值：small 比 big 最多能差多少还算“勉强可以”
    high_margin=0.0,        # 高档阈值：small >= big_s + high_margin 视为“很好”
    chunk_size=32,          # 仅对中档样本做 chunk 级 verify
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = [f"{config.system_prompt}\n{p}" for p in x["problem"]]
    batch_size = len(prompts)

    # 每个样本最终要返回的 completions / scores
    final_completions = [[] for _ in range(batch_size)]
    final_scores = [[] for _ in range(batch_size)]
    final_tokens = [[] for _ in range(batch_size)]
    final_pred = [None] * batch_size

    # 记录时间（batched=True 时要返回 list）
    t_big_gen = 0.0
    t_small_gen = 0.0
    t_prm = 0.0

    # ---------------------------
    # Stage 1: big_model warm-up
    # ---------------------------
    t0 = time.time()
    warmup_outputs = []
    for prompt in prompts:
        inputs = big_tok(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = big_model.generate(
                **inputs,
                max_new_tokens=warmup_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                num_return_sequences=1,
            )
        warmup_outputs.append(out[0])
    t1 = time.time()
    t_big_gen += (t1 - t0)

    warmup_texts = [
        big_tok.decode(ids, skip_special_tokens=True)
        for ids in warmup_outputs
    ]

    # ---------------------------
    # Stage 1 scoring: baseline
    # ---------------------------
    t2 = time.time()
    big_scores_raw = prm.score(x["problem"], [[txt] for txt in warmup_texts])
    t3 = time.time()
    t_prm += (t3 - t2)

    big_baseline_scores = [
        aggregate_scores(score_list[0], config.agg_strategy)
        for score_list in big_scores_raw
    ]

    # ------------------------------------------------------
    # Stage 2: small_model 完整生成一版，评估整体表现
    # ------------------------------------------------------
    t4 = time.time()
    small_cont_texts = []
    small_cont_tokens = []

    for warmup_txt in warmup_texts:
        inputs = small_tok(warmup_txt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = small_model.generate(
                **inputs,
                max_new_tokens=config.max_small_model_len - warmup_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
                num_return_sequences=1,
            )
        cont_ids = out[0]
        cont_txt = small_tok.decode(cont_ids, skip_special_tokens=True)
        small_cont_texts.append(cont_txt)
        small_cont_tokens.append(cont_ids.shape[-1])

    t5 = time.time()
    t_small_gen += (t5 - t4)

    small_full_texts = [
        warmup_texts[i] + "\n" + small_cont_texts[i]
        for i in range(batch_size)
    ]

    t6 = time.time()
    small_scores_raw = prm.score(x["problem"], [[txt] for txt in small_full_texts])
    t7 = time.time()
    t_prm += (t7 - t6)

    small_scores = [
        aggregate_scores(score_list[0], config.agg_strategy)
        for score_list in small_scores_raw
    ]

    # ------------------------------------------------------
    # 三档分类：
    #   good_small: small 表现非常好 → Stage3 直接用 small 全段（只总验证一次）
    #   mid_small:  small 跟 big 还算接近 → Stage3 用 chunk-level verify
    #   bad_small:  small 明显不行 → Stage3 直接交给 big
    # ------------------------------------------------------
    tier = [""] * batch_size
    for i in range(batch_size):
        big_s = big_baseline_scores[i]
        small_s = small_scores[i]

        if small_s >= big_s + high_margin:
            tier[i] = "good_small"
        elif small_s >= big_s - score_delta:
            tier[i] = "mid_small"
        else:
            tier[i] = "bad_small"

    # 先把 good_small / bad_small 直接处理掉，只给 mid_small 做 chunk 级 verify
    # ------------------------------------------------------
    # Stage 3A: good_small —— 直接采纳 small 全段输出
    # ------------------------------------------------------
    for i in range(batch_size):
        if tier[i] == "good_small":
            final_completions[i] = [small_full_texts[i]]
            final_scores[i] = [small_scores_raw[i][0]]  # 原始 PRM 分布
            final_tokens[i] = [small_cont_tokens[i]]
            final_pred[i] = small_full_texts[i]

    # ------------------------------------------------------
    # Stage 3B: bad_small —— 完全用 big_model 重新生成
    # ------------------------------------------------------
    t8 = time.time()
    for i in range(batch_size):
        if tier[i] == "bad_small":
            inputs = big_tok(warmup_texts[i], return_tensors="pt").to(device)
            with torch.inference_mode():
                out = big_model.generate(
                    **inputs,
                    max_new_tokens=config.max_model_len - warmup_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                )
            full_ids = out[0]
            full_txt = big_tok.decode(full_ids, skip_special_tokens=True)

            # 用整段 big 结果打分
            t9 = time.time()
            full_score_raw = prm.score([x["problem"][i]], [[full_txt]])
            t10 = time.time()
            t_prm += (t10 - t9)

            final_completions[i] = [full_txt]
            final_scores[i] = [full_score_raw[0][0]]
            final_tokens[i] = [full_ids.shape[-1]]
            final_pred[i] = full_txt
    t_big_gen += (time.time() - t8)

    # ------------------------------------------------------
    # Stage 3C: mid_small —— chunk 级 speculative verify
    #    思路：
    #      从 warmup_texts[i] 起步：
    #      循环：
    #         small 生成 chunk_size
    #         PRM 打分与 big_baseline 对比
    #           若 OK → 接受 small chunk
    #           若不 OK → fallback big 生成同样大小 chunk
    #                     并更新 baseline（big 的 chunk 分数）
    # ------------------------------------------------------
    max_new_tokens = config.max_model_len - warmup_tokens
    max_chunks = max_new_tokens // chunk_size

    for i in range(batch_size):
        if tier[i] != "mid_small":
            continue

        cur_text = warmup_texts[i]
        baseline = big_baseline_scores[i]

        chunk_scores = []

        for k in range(max_chunks):
            # 阶段性 delta，可以更细化（early/mid/late）
            if k < 4:
                delta = score_delta * 0.6   # 前期严格一点
            elif k < 10:
                delta = score_delta         # 中段正常
            else:
                delta = score_delta * 1.5   # 后段放松一点

            # small 尝试一个 chunk
            t_small_start = time.time()
            inputs_small = small_tok(cur_text, return_tensors="pt").to(device)
            with torch.inference_mode():
                out_small = small_model.generate(
                    **inputs_small,
                    max_new_tokens=chunk_size,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=True,
                    num_return_sequences=1,
                )
            t_small_gen += time.time() - t_small_start

            small_chunk_txt = small_tok.decode(out_small[0], skip_special_tokens=True)

            # 仅对这个 chunk（或者 cur_text+chunk，都可以）做 PRM
            t_prm_start = time.time()
            small_chunk_score_raw = prm.score(
                [x["problem"][i]],
                [[small_chunk_txt]],
            )
            t_prm += time.time() - t_prm_start

            small_chunk_score = aggregate_scores(
                small_chunk_score_raw[0][0],
                config.agg_strategy
            )

            # 判断 small 的 chunk 是否可以接受
            if small_chunk_score >= baseline - delta:
                # 接受 small chunk
                cur_text = cur_text + " " + small_chunk_txt
                chunk_scores.append(small_chunk_score)
            else:
                # fallback：用 big_model 生成同样规模的 chunk
                t_big_start = time.time()
                inputs_big = big_tok(cur_text, return_tensors="pt").to(device)
                with torch.inference_mode():
                    out_big = big_model.generate(
                        **inputs_big,
                        max_new_tokens=chunk_size,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        do_sample=True,
                        num_return_sequences=1,
                    )
                t_big_gen += time.time() - t_big_start

                big_chunk_txt = big_tok.decode(out_big[0], skip_special_tokens=True)

                # 用 big chunk 更新 baseline（big 更可信）
                t_prm_start2 = time.time()
                big_chunk_score_raw = prm.score(
                    [x["problem"][i]],
                    [[big_chunk_txt]],
                )
                t_prm += time.time() - t_prm_start2

                big_chunk_score = aggregate_scores(
                    big_chunk_score_raw[0][0],
                    config.agg_strategy
                )
                baseline = big_chunk_score
                chunk_scores.append(big_chunk_score)
                cur_text = cur_text + " " + big_chunk_txt

        # mid_small 的最终结果
        final_pred[i] = cur_text
        final_completions[i] = [cur_text]
        # 这里简单用最后 baseline 或 chunk_scores 平均做 summary
        if len(chunk_scores) > 0:
            final_scores[i] = [chunk_scores]
        else:
            final_scores[i] = [[baseline]]
        final_tokens[i] = [len(cur_text.split())]

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
        "tier": tier,  # 方便你分析三种策略各自时间/效果
    }