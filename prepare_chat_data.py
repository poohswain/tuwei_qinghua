"""
prepare_chat_data.py

把 data/chat.jsonl 的完整多輪對話，展開成 TRL SFT 用的 prompt / completion 格式。

重點：先用「整段對話」切 train / validation，再各自展開。
如果先展開再切，同一段對話的前後輪會分別落在 train 與 validation，
造成 turn-level leakage，validation loss 會看起來過度樂觀。

展開方式（每個 assistant 回覆都是一筆訓練資料）：
    對話：U1 A1 U2 A2 U3 A3
    → prompt: [system, U1]                 completion: [A1]
    → prompt: [system, U1, A1, U2]         completion: [A2]
    → prompt: [system, U1, A1, U2, A2, U3] completion: [A3]

這樣模型會學到「看完前面幾輪，再接出下一句」，也就是多輪對話能力。
"""

import json
import random
from pathlib import Path

INPUT_FILE = Path("data/chat.jsonl")
OUTPUT_DIR = Path("data/sft")

VALID_RATIO = 0.1
# 最多保留幾則歷史訊息（不含 system）。我們的對話最多 4 輪 = 8 則，設 8 可以保留完整上下文。
MAX_CONTEXT_MESSAGES = 8
SEED = 42


def build_examples(conversations):
    examples = []
    for messages in conversations:
        system = [messages[0]] if messages and messages[0]["role"] == "system" else []
        dialog = [m for m in messages if m["role"] != "system"]

        for index, message in enumerate(dialog):
            if message["role"] != "assistant" or index == 0:
                continue
            history = dialog[:index][-MAX_CONTEXT_MESSAGES:]
            # 歷史必須以 user 開頭
            while history and history[0]["role"] != "user":
                history.pop(0)
            if not history or history[-1]["role"] != "user":
                continue
            examples.append({"prompt": system + history, "completion": [message]})
    return examples


def save_jsonl(rows, path):
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(f"找不到 {INPUT_FILE}，請先執行 python build_dataset.py")

    conversations = [
        json.loads(line)["messages"]
        for line in INPUT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not conversations:
        raise RuntimeError(f"{INPUT_FILE} 沒有任何對話")

    rng = random.Random(SEED)
    rng.shuffle(conversations)

    valid_size = max(1, int(len(conversations) * VALID_RATIO))
    valid_conversations = conversations[:valid_size]
    train_conversations = conversations[valid_size:]

    train_rows = build_examples(train_conversations)
    valid_rows = build_examples(valid_conversations)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(train_rows, OUTPUT_DIR / "train.jsonl")
    save_jsonl(valid_rows, OUTPUT_DIR / "valid.jsonl")

    print("=" * 72)
    print(f"總對話數：{len(conversations)}")
    print(f"Train 對話：{len(train_conversations)} → SFT examples：{len(train_rows)}")
    print(f"Valid 對話：{len(valid_conversations)} → SFT examples：{len(valid_rows)}")
    print(f"輸出：{OUTPUT_DIR.resolve()}")
    print("=" * 72)


if __name__ == "__main__":
    main()
