"""调度层（第一层）：该不该回、延迟多久、什么档位。

原则：能不调模型就不问——省算力，也省一次出戏机会。
本项目单用户单场景，"不该回"的场景很少；主要职责是建议回复延迟
（配合节奏层的"正在输入"阶段，服务端不阻塞等待）。
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass


@dataclass
class TurnPlan:
    scene_mode: str      # daily / normal / longrun / deep（由世界书引擎最终判定）
    delay_s: float       # 建议回复延迟，前端在"正在输入"阶段播放，服务端不 sleep


class Orchestrator:
    def __init__(self):
        self._last_reply_ts = 0.0

    def plan(self, user_text: str, scene_mode: str = "normal") -> TurnPlan:
        """给出本轮的延迟建议。档位以世界书扫描结果为准，这里只做透传。

        以前这里写死 "normal"、响应里却用扫描结果，同一个字段两个真源，
        前端拿到的档位和实际注入的卡会不一致。
        """
        hour = time.localtime().tm_hour
        is_night = 0 <= hour < 6
        base = random.uniform(1.2, 3.0)
        if is_night:
            base *= 1.6  # 深夜回得更慢，像困了
        if len(user_text) >= 40:
            base += 1.0  # 长消息多"想一会儿"
        self._last_reply_ts = time.time()
        return TurnPlan(scene_mode=scene_mode, delay_s=round(base, 1))
