"""
chat_utils.py

infer_lora_chat.py 與 demo_chat.py 共用的函式：載入 Base Model + LoRA adapter、產生回覆。
"""

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_dataset import SYSTEM_PROMPT

ADAPTER_DIR = "outputs/tuwei_lora"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.1
# 送進模型的歷史最多保留幾輪（一輪 = user + assistant）
MAX_HISTORY_TURNS = 8


def load_model(adapter_dir=ADAPTER_DIR):
    """
    載入 tokenizer、Base Model，再掛上 LoRA adapter。
    Base model 名稱記錄在 adapter_config.json，PeftModel 會自動讀取。
    推論時不需要 4-bit，用 BF16／FP16 載入即可（1.5B 約 3 GB VRAM）。
    """
    config = PeftConfig.from_pretrained(adapter_dir)
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        dtype=dtype,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer


def new_conversation():
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def trim_history(messages):
    """保留 system prompt 與最近 MAX_HISTORY_TURNS 輪。"""
    limit = 1 + MAX_HISTORY_TURNS * 2
    if len(messages) <= limit:
        return messages
    return [messages[0]] + messages[-MAX_HISTORY_TURNS * 2:]


def generate_reply(model, tokenizer, messages, do_sample=True):
    inputs = tokenizer.apply_chat_template(
        trim_history(messages),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]

    generate_kwargs = dict(
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        generate_kwargs.update(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
    else:
        generate_kwargs.update(do_sample=False)

    with torch.inference_mode():
        output = model.generate(**inputs, **generate_kwargs)

    return tokenizer.decode(output[0][input_length:], skip_special_tokens=True).strip()
