import pickle
import pandas as pd
import numpy as np

with open('timing_Llama-3.2-3B-Instruct_best_of_n_20251123_151925.pkl', 'rb') as f:
    timing_results = pickle.load(f)

# print(timing_results)

# print 全部字段
print("全部字段:", timing_results.columns.tolist())

'''
为啥不一样了
   llm_gen_time  prm_score_time  ...  timing_timestamp         completion_tokens
0      6.643349        4.855974  ...   20251123_123114      [311, 510, 377, 351]
1      6.643349        4.855974  ...   20251123_123114  [1152, 1865, 1073, 1451]
2      6.643349        4.855974  ...   20251123_123114      [529, 427, 452, 481]
3      6.643349        4.855974  ...   20251123_123114      [200, 217, 257, 145]
4      6.643349        4.855974  ...   20251123_123114      [368, 530, 513, 481]
5      6.643349        4.855974  ...   20251123_123114      [138, 226, 164, 282]
6      6.643349        4.855974  ...   20251123_123114      [432, 531, 602, 602]
7      6.643349        4.855974  ...   20251123_123114     [726, 770, 1102, 573]
8      6.643349        4.855974  ...   20251123_123114      [260, 267, 295, 306]
9      6.643349        4.855974  ...   20251123_123114      [593, 646, 820, 478]
'''

if isinstance(timing_results, pd.DataFrame):
    # draft_gen_time 是 draft模型生成时间，prm_score_time 是验证时间
    print(timing_results[['draft_gen_time', 'prm_score_time']].head())
    #print("draft_gen_time 平均:", timing_results['draft_gen_time'].mean())
    print("prm_score_time 平均:", timing_results['prm_score_time'].mean())
    print("全部字段:", timing_results.columns.tolist())
    print(timing_results.head())
else:
    # 如果是 list of dict
    draft_gen_time = np.array([r.get('draft_gen_time', None) for r in timing_results if isinstance(r, dict)])
    prm_score_time = np.array([r.get('prm_score_time', None) for r in timing_results if isinstance(r, dict)])
    print("draft_gen_time: 前5项:", draft_gen_time[:5])
    print("prm_score_time: 前5项:", prm_score_time[:5])
    print("draft_gen_time 平均:", draft_gen_time.mean())
    print("prm_score_time 平均:", prm_score_time.mean())
    # 打印全部字段
    if len(timing_results) > 0 and isinstance(timing_results[0], dict):
        print("全部字段:", list(timing_results[0].keys()))
        for i, item in enumerate(timing_results[:5]):
            print(f"QA {i+1}: {item}")
    else:
        print("timing_results 格式无法解析字段")