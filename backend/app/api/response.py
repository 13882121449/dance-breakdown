"""统一 API 响应构造工具。

全项目统一响应格式：``{code: 0, message: "ok", data: ...}``；
``code != 0`` 表示错误。成功由 :func:`ok` 构造，错误由 FastAPI 异常处理器
（见 main.py）转换为 ``{code: <http_status>, message, data: {}}``。
"""

from __future__ import annotations

from typing import Any

from app.config import API_CODE_OK, API_MESSAGE_OK


def ok(data: Any = None) -> dict[str, Any]:
    """构造成功响应；``data`` 缺省时返回空对象（保持结构稳定）。"""
    return {
        "code": API_CODE_OK,
        "message": API_MESSAGE_OK,
        "data": {} if data is None else data,
    }


def fail(code: int, message: str) -> dict[str, Any]:
    """构造业务错误响应（供需要显式返回错误体时使用）。"""
    return {"code": code, "message": message, "data": {}}
