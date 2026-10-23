"""记忆层：SQLite 单文件承载全部结构化数据（方案 §6）。

零额外服务进程——不引 Milvus/Qdrant/Redis。
女友场景的记忆规模是几千条，1 万条 × 512 维 ≈ 20MB，暴力比对毫秒级足够。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (          -- 对话落盘备份（滑动窗口的持久化副本）
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL,                          -- user / assistant
  content TEXT NOT NULL,
  ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS memories (          -- 中期记忆：事实/偏好/事件/承诺
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,                          -- fact / preference / event / promise
  content TEXT NOT NULL,                       -- 结构化短句（绝不存总结段落）
  detail TEXT DEFAULT '',                      -- 补充字段（倾向/到期时间/完成状态，JSON）
  source TEXT DEFAULT '',                      -- 来源对话片段，可人工核对回滚
  mentions INTEGER DEFAULT 1,
  first_ts REAL NOT NULL,
  last_ts REAL NOT NULL,
  status TEXT DEFAULT 'active'                 -- active / done / dropped
);
CREATE TABLE IF NOT EXISTS timeline (          -- 长期记忆：关系里程碑与事件线
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL NOT NULL,
  title TEXT NOT NULL,
  detail TEXT DEFAULT '',
  mood_tag TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS profile (           -- 用户画像（滚动更新，按类别聚合）
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);
"""


@dataclass
class Memory:
    id: int
    kind: str
    content: str
    detail: str
    mentions: int
    first_ts: float
    last_ts: float
    status: str


class MemoryStore:
    """三层记忆的存储实现。所有写入都带来源，方便人工审与回滚。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # FastAPI 把同步端点丢进线程池执行，连接允许跨线程使用；
        # 用锁串行化读写，避免同一连接被并发触碰（单用户场景开销可忽略）
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── 对话 ──
    def add_message(self, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (role, content, ts) VALUES (?, ?, ?)",
                (role, content, time.time()),
            )
            self._conn.commit()

    def recent_messages(self, n: int) -> list[dict]:
        """最近 n 条消息（旧 → 新），供滑动窗口与触发扫描。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, ts FROM messages ORDER BY id DESC LIMIT ?", (n,)
            ).fetchall()
        return [{"role": r, "content": c, "ts": t} for r, c, t in reversed(rows)]

    def recent_turns(self, turns: int) -> list[dict]:
        """最近 n 轮对话（一轮 = 用户一句 + 她一句，最多 2n 条消息）。

        为什么按"轮"而不是按"条"：上下文窗口的语义是"最近聊了几轮"，
        按条数写死 20 会让配置里的 recent_turns 变成没人读的死参数。
        """
        return self.recent_messages(max(1, int(turns)) * 2)

    def last_interaction_ts(self) -> float | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT ts FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    # ── 记忆 ──
    def add_memory(self, kind: str, content: str, source: str = "",
                   detail: dict | None = None) -> int:
        """写入一条记忆；同内容去重并累加提及次数。"""
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT id, mentions FROM memories WHERE content = ? AND status = 'active'",
                (content,),
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE memories SET mentions = mentions + 1, last_ts = ? WHERE id = ?",
                    (now, row[0]),
                )
                self._conn.commit()
                return row[0]
            cur = self._conn.execute(
                "INSERT INTO memories (kind, content, detail, source, first_ts, last_ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (kind, content, json.dumps(detail or {}, ensure_ascii=False), source, now, now),
            )
            self._conn.commit()
            return cur.lastrowid

    def search_memories(self, keyword: str, limit: int = 8) -> list[Memory]:
        """暴力关键词检索（万级以内毫秒级，方案 §6.2）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, content, detail, mentions, first_ts, last_ts, status "
                "FROM memories WHERE content LIKE ? AND status = 'active' "
                "ORDER BY mentions DESC, last_ts DESC LIMIT ?",
                (f"%{keyword}%", limit),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def top_memories(self, limit: int = 10) -> list[Memory]:
        """最常被提及的记忆（供主动消息与画像聚合）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, content, detail, mentions, first_ts, last_ts, status "
                "FROM memories WHERE status = 'active' "
                "ORDER BY mentions DESC, last_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def memories_by_kind(self, kind: str, limit: int = 20) -> list[Memory]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, kind, content, detail, mentions, first_ts, last_ts, status "
                "FROM memories WHERE kind = ? AND status = 'active' "
                "ORDER BY last_ts DESC LIMIT ?",
                (kind, limit),
            ).fetchall()
        return [Memory(*r) for r in rows]

    def update_memory_status(self, memory_id: int, status: str) -> None:
        """人工审后的状态流转（active / done / dropped）——抽取结果必须可回滚。"""
        with self._lock:
            self._conn.execute(
                "UPDATE memories SET status = ? WHERE id = ?", (status, memory_id)
            )
            self._conn.commit()

    # ── 时间线与画像 ──
    def add_timeline(self, title: str, detail: str = "", mood_tag: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO timeline (ts, title, detail, mood_tag) VALUES (?, ?, ?, ?)",
                (time.time(), title, detail, mood_tag),
            )
            self._conn.commit()

    def recent_timeline(self, limit: int = 5) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, title, detail, mood_tag FROM timeline ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"ts": r[0], "title": r[1], "detail": r[2], "mood_tag": r[3]} for r in reversed(rows)]

    def set_profile(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO profile (key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (key, value, time.time()),
            )
            self._conn.commit()

    def get_profile(self, key: str, default: str = "") -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM profile WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else default

    def all_profile(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM profile").fetchall()
        return dict(rows)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
