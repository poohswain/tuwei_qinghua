"""
train_qlora.py

用 4-bit QLoRA 微調 Qwen2.5-Instruct，建立可以多輪對話的「土味情話專家」。

流程：
    data/sft/train.jsonl、valid.jsonl（prompt / completion 格式）
        ↓
    Base model 以 4-bit NF4 載入（bitsandbytes）
        ↓
    prepare_model_for_kbit_training()
        ↓
    加上 LoRA adapter（只訓練少量參數）
        ↓
    TRL SFTTrainer（completion_only_loss：只對專家的回覆計算 loss）
        ↓
    每個 epoch 評估一次、early stopping、載入 validation loss 最低的 Best Model
        ↓
    outputs/tuwei_lora/（LoRA adapter + tokenizer）

跟課堂範例的差異：
    1. 預設模型改成 Qwen2.5-1.5B-Instruct（0.5B 講笑話的能力比較弱）。
    2. 自動判斷精度：RTX 30 系列以上用 BF16，較舊的 GPU 用 FP16。
    3. 語料只有幾百筆，改成「每個 epoch」評估與存檔，並提高 learning rate。
    4. checkpoint-* 存到 checkpoints/，Best Model 另外存到 outputs/，方便上傳 GitHub。
"""

from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, set_seed
from trl import SFTConfig, SFTTrainer

# =============================================================================
# 設定
# =============================================================================

# GPU 記憶體 6 GB 以下可改成 "Qwen/Qwen2.5-0.5B-Instruct"；12 GB 以上可試 "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_DIR = Path("data/sft")
CHECKPOINT_DIR = Path("checkpoints/tuwei_lora")
OUTPUT_DIR = Path("outputs/tuwei_lora")

MAX_LENGTH = 1024
EPOCHS = 10
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
GRADIENT_CHECKPOINTING = True
WARMUP_STEPS = 10
OPTIM = "paged_adamw_8bit"
LOGGING_STEPS = 5
EARLY_STOPPING_PATIENCE = 2
SAVE_TOTAL_LIMIT = 2

# 只對 completion（專家的回覆）計算 loss，不學使用者說的話
COMPLETION_ONLY_LOSS = True
# Qwen2.5 的對話結束符號
EOS_TOKEN = "<|im_end|>"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = "all-linear"

SEED = 42


# =============================================================================
# 輔助函式
# =============================================================================

def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("找不到 CUDA GPU。QLoRA 需要 NVIDIA GPU，請確認已安裝 CUDA 版的 PyTorch。")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    use_bf16 = torch.cuda.is_bf16_supported()

    print("=" * 72)
    print(f"GPU：{name}（capability {capability}，{total_gb:.1f} GB）")
    print(f"PyTorch：{torch.__version__}，CUDA：{torch.version.cuda}")
    print(f"Mixed precision：{'BF16' if use_bf16 else 'FP16'}")
    print("=" * 72)
    return use_bf16


def cast_trainable_parameters_to_fp32(model):
    """LoRA 參數很少，用 FP32 更新比較穩定，也避免 FP16 GradScaler 遇到 BF16 參數時出錯。"""
    for param in model.parameters():
        if param.requires_grad and param.dtype != torch.float32:
            param.data = param.data.to(torch.float32)


def print_trainable_dtypes(model):
    dtypes = Counter(str(p.dtype) for p in model.parameters() if p.requires_grad)
    print("Trainable parameter dtypes：" + "、".join(f"{k} × {v}" for k, v in sorted(dtypes.items())))


# =============================================================================
# 主程式
# =============================================================================

def main():
    use_bf16 = check_gpu()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    set_seed(SEED)

    train_file = DATA_DIR / "train.jsonl"
    valid_file = DATA_DIR / "valid.jsonl"
    for path in (train_file, valid_file):
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {path}，請先執行 python build_dataset.py 與 python prepare_chat_data.py")

    data = load_dataset("json", data_files={"train": str(train_file), "validation": str(valid_file)})
    print(data)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_config,
        dtype=compute_dtype,
        device_map={"": 0},
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    cast_trainable_parameters_to_fp32(model)
    model.print_trainable_parameters()
    print_trainable_dtypes(model)

    args = SFTConfig(
        output_dir=str(CHECKPOINT_DIR),
        max_length=MAX_LENGTH,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=not use_bf16,
        bf16=use_bf16,
        optim=OPTIM,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        completion_only_loss=COMPLETION_ONLY_LOSS,
        eos_token=EOS_TOKEN,
        report_to="none",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=data["train"],
        eval_dataset=data["validation"],
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    # SFTTrainer 初始化時可能又調整過 dtype，再確認一次
    cast_trainable_parameters_to_fp32(trainer.model)

    trainer.train()

    print(f"最佳 checkpoint：{trainer.state.best_model_checkpoint}")
    print(f"最佳 eval_loss：{trainer.state.best_metric}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    trainer.state.save_to_json(str(OUTPUT_DIR / "trainer_state.json"))
    print(f"LoRA adapter 已儲存：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
