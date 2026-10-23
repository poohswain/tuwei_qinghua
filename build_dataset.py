"""
build_dataset.py

把 corpus/*.txt 的多輪對話語料，轉成統一的 JSONL 格式（取代課堂範例的 download_chat_data.py）。

語料格式（corpus/*.txt）：
    你：教我一句早安的情話。
    專家：早安，今天的太陽……
    你：她回我……
    專家：你就說……
                           ← 空一行，代表下一段對話
    你：……
    專家：……

    以 # 開頭的行是註解。

輸出（data/chat.jsonl，一行一段完整對話）：
    {"messages": [
        {"role": "system", "content": "你是一個情話高手……"},
        {"role": "user", "content": "教我一句早安的情話。"},
        {"role": "assistant", "content": "早安，今天的太陽……"},
        ...
    ]}
"""

import json
from collections import Counter
from pathlib import Path

CORPUS_DIR = Path("corpus")
OUTPUT_FILE = Path("data/chat.jsonl")

SYSTEM_PROMPT = (
    "你是一個土味情話高手，你會以親切、溫和、幽默的口吻回答使用者的問題，"
    "教使用者在不同情境說出又土又甜的情話，並根據對方的反應接出下一句。"
    "請使用台灣繁體中文，回答簡短自然。"
)

USER_PREFIXES = ("你：", "你:")
ASSISTANT_PREFIXES = ("專家：", "專家:")

# 每段對話至少要有幾輪（一輪 = user + assistant）
MIN_ROUNDS = 2


def parse_line(line):
    for prefix in USER_PREFIXES:
        if line.startswith(prefix):
            return "user", line[len(prefix):].strip()
    for prefix in ASSISTANT_PREFIXES:
        if line.startswith(prefix):
            return "assistant", line[len(prefix):].strip()
    return None, None


def parse_file(path: Path):
    """讀一個語料檔，回傳 list[list[message]]。"""
    conversations = []
    current = []

    def flush():
        if current:
            conversations.append(list(current))
            current.clear()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line.startswith("#"):
            continue
        if not line:
            flush()
            continue

        role, content = parse_line(line)
        if role is None:
            # 沒有前綴的行視為上一句的換行延續
            if not current:
                raise ValueError(f"{path}:{line_number} 無法辨識的行：{line}")
            current[-1]["content"] += "\n" + line
            continue
        if not content:
            raise ValueError(f"{path}:{line_number} 內容是空的")
        current.append({"role": role, "content": content, "_line": line_number})

    flush()
    return conversations


def validate(conversation, path):
    """檢查 user / assistant 交錯出現，以 user 開頭、assistant 結尾。"""
    start = conversation[0]["_line"]
    roles = [m["role"] for m in conversation]
    expected = ["user", "assistant"] * (len(roles) // 2)
    if roles != expected or len(roles) % 2 != 0:
        raise ValueError(f"{path}:{start} 這段對話的「你／專家」沒有交錯出現，或沒有以「專家」結尾：{roles}")
    rounds = len(roles) // 2
    if rounds < MIN_ROUNDS:
        raise ValueError(f"{path}:{start} 這段對話只有 {rounds} 輪，至少要 {MIN_ROUNDS} 輪")
    return rounds


def main():
    files = sorted(CORPUS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"{CORPUS_DIR} 裡沒有任何 .txt 語料")

    rows = []
    rounds_counter = Counter()
    seen_openers = set()

    for path in files:
        conversations = parse_file(path)
        for conversation in conversations:
            rounds = validate(conversation, path)
            rounds_counter[rounds] += 1

            opener = conversation[0]["content"]
            if opener in seen_openers:
                print(f"提醒：開場白重複 → {opener}（{path.name}）")
            seen_openers.add(opener)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            messages += [{"role": m["role"], "content": m["content"]} for m in conversation]
            rows.append({"messages": messages})
        print(f"{path.name}: {len(conversations)} 段對話")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_rounds = sum(r * c for r, c in rounds_counter.items())
    print("=" * 72)
    print(f"總對話數：{len(rows)}")
    print(f"總輪數（= assistant 回覆數）：{total_rounds}")
    print("每段對話的輪數分佈：" + "、".join(f"{r} 輪 × {c}" for r, c in sorted(rounds_counter.items())))
    print(f"輸出：{OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
