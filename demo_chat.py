"""
demo_chat.py

用固定的使用者問題跑多輪對話，並比較「微調前（Base Model）」與「微調後（LoRA）」的回覆，
結果存到 results/demo_chat.md，可以直接貼進 README 當作成果展示。

DEMO_SCRIPTS 刻意選用「訓練語料裡沒有出現過」的情境，用來觀察模型是否真的學會風格，
而不是背答案。可以自行增減。
"""

from pathlib import Path

from transformers import set_seed

from chat_utils import generate_reply, load_model, new_conversation

RESULT_FILE = Path("results/demo_chat.md")
SEED = 42

DEMO_SCRIPTS = {
    "在籃球場搭訕": [
        "我在籃球場看到一個很會投三分球的女生，要怎麼搭話？",
        "她說「你也會打球嗎？」",
        "她說「那我們來比一場，輸的請飲料」",
        "她贏了，說「快去買飲料」",
    ],
    "跟同事撩一下": [
        "我想對同部門的同事說一句土味情話，但不要太油。",
        "她問「你今天怎麼怪怪的？」",
        "她笑著說「你是不是想約我吃飯？」",
    ],
    "宿舍停電": [
        "宿舍突然停電了，我想傳訊息給喜歡的人，要說什麼？",
        "她回「我這邊也停電了，好黑喔」",
        "她說「那你要來陪我嗎？」",
    ],
}


def run_script(model, tokenizer, user_turns):
    messages = new_conversation()
    for text in user_turns:
        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": generate_reply(model, tokenizer, messages)})
    return messages


def to_markdown(title, messages):
    lines = [f"#### {title}", ""]
    for message in messages[1:]:
        speaker = "你" if message["role"] == "user" else "專家"
        lines.append(f"**{speaker}：** {message['content']}  ")
    lines.append("")
    return lines


def main():
    print("正在載入 Base Model + LoRA adapter...")
    model, tokenizer = load_model()

    output = ["# 多輪對話成果：微調前 vs 微調後", ""]
    for name, user_turns in DEMO_SCRIPTS.items():
        print("=" * 72)
        print(f"情境：{name}")

        set_seed(SEED)
        tuned = run_script(model, tokenizer, user_turns)

        # disable_adapter()：暫時關掉 LoRA，等於使用原始的 Base Model
        set_seed(SEED)
        with model.disable_adapter():
            base = run_script(model, tokenizer, user_turns)

        for message in tuned[1:]:
            speaker = "你" if message["role"] == "user" else "專家"
            print(f"{speaker}：{message['content']}")

        output.append(f"## {name}")
        output.append("")
        output += to_markdown("微調後（LoRA）", tuned)
        output += to_markdown("微調前（Base Model）", base)

    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.write_text("\n".join(output), encoding="utf-8")
    print("=" * 72)
    print(f"已儲存：{RESULT_FILE.resolve()}")


if __name__ == "__main__":
    main()
