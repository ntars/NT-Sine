"""生成层：OpenAI 兼容客户端，接 mlx-lm / LM Studio 推理服务（方案 §8.3）。

走 /v1/chat/completions——这层格式是 ai-companion-kit 等生态项目能直连的钥匙。
采样参数集中在此，温度硬顶 0.80（源于 1.5B 时代实测；4B 基座下是否放宽待 v6 验收后实验），活泼度交给 top_p。
"""

from __future__ import annotations

import httpx

from app.config import SETTINGS


class GenerationError(RuntimeError):
    """模型不可用或返回异常。"""


class Generator:
    def __init__(self):
        self.cfg = SETTINGS["ollama"]
        self.sampling = SETTINGS["sampling"]
        self.base_url = self.cfg["base_url"].rstrip("/")
        self.model = self.cfg["model"]

    def chat(self, messages: list[dict], num_ctx: int | None = None,
             max_tokens: int | None = None) -> str:
        """一次对话补全。messages 为 OpenAI 格式 [{role, content}]。"""
        s = self.sampling
        temperature = min(0.80, float(s["temperature"]))  # 硬顶，防"试错式调参"越界
        # 同一份参数写两处：
        #   options  → Ollama 原生字段（num_ctx/num_predict 只有 Ollama 认）
        #   顶层标准字段 → mlx_lm.server / 其他 OpenAI 兼容后端只认这些；
        #   缺了它们会退化成贪心解码（实测表现：原样复述用户的话）
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "top_p": float(s["top_p"]),
            "max_tokens": int(max_tokens or s["max_tokens"]),
            "presence_penalty": float(s["presence_penalty"]),
            "frequency_penalty": float(s["frequency_penalty"]),
            "options": {
                "temperature": temperature,
                "top_p": float(s["top_p"]),
                "top_k": int(s["top_k"]),
                "repeat_penalty": float(s["repeat_penalty"]),
                "repeat_last_n": int(s["repeat_last_n"]),
                "num_ctx": int(num_ctx or SETTINGS["context"]["num_ctx"]),
                "num_predict": int(max_tokens or s["max_tokens"]),
            },
        }
        try:
            with httpx.Client(timeout=float(self.cfg["timeout_s"])) as client:
                resp = client.post(f"{self.base_url}/v1/chat/completions", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPError as e:
            raise GenerationError(f"调用模型服务失败：{e}") from e
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise GenerationError(f"模型服务返回结构异常：{data}") from e
        # 服务端偶发返回 200 OK 但 content 为 null（模型空回复），
        # 直接 content.strip() 会抛 AttributeError 导致接口 500；
        # 统一转 GenerationError，由调用方回落人设兜底句
        if not isinstance(content, str) or not content.strip():
            raise GenerationError("模型返回空回复（content 为 null 或空白）")
        return content.strip()

    def chat_json(self, messages: list[dict], num_ctx: int = 2048) -> dict | list | None:
        """轻量抽取用：要求 JSON 输出，失败返回 None（绝不抛错打断主链路）。"""
        import json
        import re
        try:
            text = self.chat(messages, num_ctx=num_ctx, max_tokens=400)
        except GenerationError:
            return None
        match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
