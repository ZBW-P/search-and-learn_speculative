from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from datasets import load_dataset

def LLM_compressor(MODEL_ID=None, OUTPUT=None):

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")
    ds = ds.shuffle(seed=42)

    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,add_special_tokens=False,)}

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(sample["text"], max_length=2048, truncation=True)

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"]),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=2048,
        num_calibration_samples=512,
        output_dir=OUTPUT,
    )
    print(f"Compressed model saved to {OUTPUT}")
if __name__ == "__main__":
    LLM_compressor(
        MODEL_ID="meta-llama/Llama-3.2-3B-Instruct",
        OUTPUT="./meta-llama/Llama-3.2-3B-Instruct-GPTQ-8bit"
    )