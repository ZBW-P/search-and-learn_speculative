import copy
import logging
import numpy as np
from tqdm import tqdm
from vllm import LLM, SamplingParams

from sal.config import Config
from sal.models.reward_models import PRM
from sal.utils.score import aggregate_scores
from .utils import Beam, build_conv, generate_k_steps

logger = logging.getLogger()


def dual_beam_search(
    examples,
    config: Config,
    big_llm: LLM,
    small_llm: LLM,
    prm: PRM,
):
    problems = examples["problem"]
    beams = _dual_beam_search_internal(problems, config, big_llm, small_llm, prm)

    from collections import defaultdict
    grouped = defaultdict(list)
    for beam in beams:
        grouped[beam.prompt].append(beam)

    results = {"completions": [], "pred": [], "completion_tokens": [], "scores": []}

    for p in problems:
        group = grouped[p]
        completions = [b.current_text for b in group]
        agg_scores = [
            aggregate_scores(b.all_scores, config.agg_strategy) for b in group
        ]
        pred = completions[np.argmax(agg_scores)]

        results["completions"].append(completions)
        results["scores"].append([b.all_scores for b in group])
        results["pred"].append(pred)
        results["completion_tokens"].append([b.completion_tokens for b in group])

    return results


def _dual_beam_search_internal(
    batch_of_prompts,
    config: Config,
    big_llm: LLM,
    small_llm: LLM,
    prm: PRM,
) -> list[Beam]:

    BIG_WARMUP_ITERS = getattr(config, "big_warmup_iters", 2)
    SWITCH_THRESHOLD = getattr(config, "switch_prm_threshold", 0.20)
    FALLBACK_THRESHOLD = getattr(config, "fallback_prm_threshold", 0.10)

    use_small = False

    beams = []
    for p in batch_of_prompts:
        for i in range(config.n):
            beams.append(
                Beam(
                    prompt=p,
                    index=i,
                    current_text="",
                    next_texts=None,
                    lookahead_texts=None,
                    pruned=False,
                    completed=False,
                    stop_reasons=None,
                    history=[],
                    best_scores=[],
                    all_scores=[],
                    previous_text=None,
                    completion_tokens=0,
                )
            )

    completed_beams = []

    for depth in tqdm(range(config.num_iterations), desc="Dual Beam Search"):

        if depth == 0:
            active = [b for b in beams if not b.pruned]
        else:
            active = [b for b in active if not b.pruned]

        if len(active) == 0:
            break

        if len(active) != config.n:
            rep = (config.n // len(active)) + 1
            active = (active * rep)[: config.n]

        if depth < BIG_WARMUP_ITERS:
            cur_llm = big_llm
        elif use_small:
            cur_llm = small_llm
        else:
            cur_llm = big_llm

        tokenizer = cur_llm.get_tokenizer()
        if config.custom_chat_template is not None:
            tokenizer.chat_template = config.custom_chat_template

        convs = [
            build_conv(b.prompt, b.current_text, config.system_prompt)
            for b in active
        ]

        continue_msg = depth > 0
        add_gen_prompt = depth == 0

        templated = tokenizer.apply_chat_template(
            convs,
            tokenize=False,
            add_generation_prompt=add_gen_prompt,
            continue_final_message=continue_msg,
        )

        if depth == config.num_iterations - 1:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                max_tokens=config.max_tokens,
                n=1,
            )
            lookahead = 0
        else:
            sampling_params = SamplingParams(
                temperature=config.temperature,
                top_p=config.top_p,
                stop=["\n\n"],
                include_stop_str_in_output=True,
                max_tokens=config.max_tokens,
                n=1,
            )
            lookahead = config.lookahead

        gen_results = generate_k_steps(
            templated, lookahead, cur_llm, sampling_params, 1
        )

        prompts, completions = [], []
        for beam, gen in zip(active, gen_results, strict=True):
            beam.next_texts = gen.next_texts
            beam.stop_reasons = gen.stop_reasons
            beam.lookahead_texts = gen.lookahead_texts
            beam.completion_tokens += gen.completion_tokens

            step_text = gen.next_texts[0]
            beam.current_text += step_text
            beam.history.append(step_text)

            if (
                gen.stop_reasons[0] in ["EOS", "length"]
                or step_text == ""
            ):
                beam.completed = True
                completed_beams.append(beam)

            prompts.append(beam.prompt)
            completions.append([beam.current_text])
            
        scores = prm.score(prompts, completions)

        agg_scores = [
            [aggregate_scores(s, config.agg_strategy) for s in score]
            for score in scores
        ]
        for b, sc in zip(active, scores, strict=True):
            b.all_scores = sc[0]

        agg_scores = [
            agg_scores[i] for i, b in enumerate(active) if not b.completed
        ]
        active = [b for b in active if not b.completed]

        if len(active) == 0:
            break

        # =============== 策略 1 ：切换到小模型 ===============
        flat_scores = np.array(agg_scores).flatten()
        avg_score = float(flat_scores.mean())

        if (not use_small) and (depth >= BIG_WARMUP_ITERS):
            if avg_score > SWITCH_THRESHOLD:
                use_small = True

        # =============== 策略 2：小模型生成太差 → fallback 大模型 ===============
        if use_small and (avg_score < FALLBACK_THRESHOLD):
            use_small = False

        # =============== 去重 + top-k 剪枝 ===============
        if config.filter_duplicates:
            uniq = {}
            for i_b, b in enumerate(active):
                if b.current_text not in uniq:
                    uniq[b.current_text] = i_b
            active = [active[i_b] for i_b in uniq.values()]
            agg_scores = [agg_scores[i_b] for i_b in uniq.values()]

        top_idx = np.argsort(np.array(agg_scores).flatten())[
            -(config.n // config.beam_width) :
        ]
        for idx, b in enumerate(active):
            if idx not in top_idx:
                b.pruned = True

    # =============== 输出 top beams ===============
    # Filter completed beams for those with top config.n scores
    if config.sort_completed:
        completed_beams = sorted(
            completed_beams,
            key=lambda b: aggregate_scores(b.all_scores, config.agg_strategy),
            reverse=True,
        )[: config.n]
    else:
        completed_beams = completed_beams[: config.n]

    if len(completed_beams) != config.n:
        # If we don't have enough completed_beams, duplicate until we reach config.n
        repeats = (config.n // len(completed_beams)) + 1
        logger.debug(
            f"Extending completed_beams with {repeats} repetitions to reach size {config.n}"
        )
        extended_completed_beams = [
            copy.deepcopy(b) for b in (completed_beams * repeats)[: config.n]
        ]
        completed_beams = extended_completed_beams

    return completed_beams
