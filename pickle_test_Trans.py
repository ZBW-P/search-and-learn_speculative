import pickle
import numpy as np
import pandas as pd
from pprint import pprint
from datasets import Dataset
# ========= 1. 读 pkl ==========
pkl_path = "/scratch/qh2262/search-and-learn_speculative/pkl_results/timing_Llama-3.2-3B-Instruct_best_of_n_speculative_transformers_20251125_162438.pkl"
print("=== 尝试 pickle.load 读取 ===")
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

print(f"读取类型: {type(data)}")

# ========== 情况 1：它本来就是 HuggingFace Dataset ==========
if isinstance(data, Dataset):
    print("\n=== 这是 HuggingFace Dataset！===")
    print(data)

    print("\n=== 字段 ===")
    print(data.column_names)

    # 找所有包含 "time" 的字段
    time_cols = [c for c in data.column_names if "time" in c.lower()]
    print("\n=== 检测到时间字段 ===")
    print(time_cols)

    # 每个时间字段做统计
    for col in time_cols:
        arr = np.array(data[col])

        # 如果是 list-of-list → 则取平均值
        if arr.dtype == object:
            arr = np.array([np.mean(x) if isinstance(x, list) else x for x in arr])

        print(f"{col}: mean={arr.mean():.4f}, min={arr.min():.4f}, max={arr.max():.4f}")

    # completion token length
    if "completion_tokens" in data.column_names:
        ct = np.array([np.mean(toks) for toks in data["completion_tokens"]])
        print("\n=== completion_tokens 长度统计 ===")
        print(f"mean={ct.mean():.2f}, min={ct.min():.2f}, max={ct.max():.2f}")

else:
    print("⚠️ 不是 HF Dataset，而是：", type(data))
    print("无法解析此格式，请确认保存方式")