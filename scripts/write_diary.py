#!/usr/bin/env python3
"""手动写一篇日记（调试用）：python scripts/write_diary.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import SETTINGS
from app.diary.author import write_diary
from app.generation.client import Generator
from app.memory.store import MemoryStore


def main() -> int:
    store = MemoryStore(SETTINGS["app"]["db_path"])
    memory_text = "；".join(m.content for m in store.top_memories(6))
    entry = write_diary(Generator(), store, memory_text=memory_text)
    if entry is None:
        print("未写出日记（今天已写过，或模型不可用）")
        return 1
    print(f"标题：{entry['title']}")
    print(f"情绪：{entry['mood']}")
    print(f"正文：{entry['text']}")
    store.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
