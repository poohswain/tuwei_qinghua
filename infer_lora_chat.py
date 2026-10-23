"""
infer_lora_chat.py

跟微調後的「土味情話專家」進行多輪互動式對話。
每次回答都會把前面的對話歷史一起送進模型，所以可以接續上一輪的內容。

指令：
    /reset    清除對話歷史
    /history  顯示目前的對話歷史
    /save     把目前對話存成 results/chat_時間.md（放 README 用）
    exit / quit / q  結束
"""

from datetime import datetime
from pathlib import Path

import torch

from chat_utils import generate_reply, load_model, new_conversation

RESULT_DIR = Path("results")


def save_transcript(messages):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / f"chat_{datetime.now():%Y%m%d_%H%M%S}.md"
    lines = ["# 土味情話專家對話紀錄", ""]
    for message in messages[1:]:
        speaker = "你" if message["role"] == "user" else "專家"
        lines.append(f"**{speaker}：** {message['content']}  ")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    print("正在載入 Base Model + LoRA adapter...")
    model, tokenizer = load_model()
    messages = new_conversation()

    print("=" * 72)
    print("土味情話專家已上線！輸入訊息後按 Enter 開始對話。")
    print("指令：/reset 清除歷史、/history 顯示歷史、/save 存檔、q 結束")
    print("=" * 72)

    while True:
        user_text = input("\n你：").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit", "q"}:
            print("\n專家：下次見，記得把今天學的情話用在對的人身上！")
            break
        if user_text == "/reset":
            messages = new_conversation()
            print("（已清除對話歷史）")
            continue
        if user_text == "/history":
            for index, message in enumerate(messages[1:], start=1):
                speaker = "你" if message["role"] == "user" else "專家"
                print(f"[{index}] {speaker}：{message['content']}")
            continue
        if user_text == "/save":
            print(f"（已存檔：{save_transcript(messages)}）")
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            reply = generate_reply(model, tokenizer, messages)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            messages.pop()
            print("（GPU 記憶體不足，請輸入 /reset 清除歷史後再試）")
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"\n專家：{reply}")


if __name__ == "__main__":
    main()
