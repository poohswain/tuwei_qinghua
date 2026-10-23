"""
show_versions.py

列出 README「安裝套件」需要的版本號，輸出成 Markdown 條列，直接複製貼上即可。
"""

from importlib.metadata import PackageNotFoundError, version

PACKAGES = [
    "torch",
    "transformers",
    "trl",
    "peft",
    "bitsandbytes",
    "datasets",
    "accelerate",
    "matplotlib",
]

for name in PACKAGES:
    try:
        print(f"- {name} ({version(name)})")
    except PackageNotFoundError:
        print(f"- {name} (未安裝)")
