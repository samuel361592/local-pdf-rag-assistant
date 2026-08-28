"""Ollama vision-language model integration."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field, ValidationError

from .config import Settings
from .visual_prompt import VISUAL_SYSTEM_PROMPT, build_visual_user_prompt


class VisualAnalysisResult(BaseModel):
    page_type: str = Field(description="頁面或圖片類型")
    summary: str = Field(description="一至三句整體摘要")
    visible_text: list[str] = Field(default_factory=list)
    visual_facts: list[str] = Field(default_factory=list)
    table_content: list[str] = Field(default_factory=list)
    chart_insights: list[str] = Field(default_factory=list)
    abnormal_visuals: list[str] = Field(default_factory=list)
    uncertain_items: list[str] = Field(default_factory=list)


class VLMError(RuntimeError):
    """代表不含圖片或模型原始回應的安全 VLM 錯誤。"""


def _check_ollama_version(settings: Settings) -> str:
    request = Request(
        f"{settings.ollama_base_url.rstrip('/')}/api/version",
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=settings.vlm_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        version = str(payload.get("version", "")).strip()
        if not version:
            raise ValueError("missing version")
        return version
    except (
        HTTPError,
        URLError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        AttributeError,
        TypeError,
        ValueError,
    ) as exc:
        raise VLMError(
            "無法確認 Ollama 版本，請確認服務已啟動且 OLLAMA_BASE_URL 設定正確。"
        ) from exc


def _create_vlm_model(settings: Settings) -> ChatOllama:
    try:
        _check_ollama_version(settings)
        return ChatOllama(
            model=settings.vlm_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            num_predict=settings.vlm_num_predict,
            validate_model_on_init=True,
            sync_client_kwargs={"timeout": settings.vlm_timeout_seconds},
        )
    except VLMError:
        raise
    except Exception as exc:
        lowered = str(exc).casefold()
        if "not found" in lowered or "404" in lowered:
            raise VLMError(
                f"找不到 VLM 模型 {settings.vlm_model}，請先執行："
                f"ollama pull {settings.vlm_model}"
            ) from exc
        raise VLMError(
            "無法連線到 Ollama 以啟用 VLM，請確認服務已啟動且位址設定正確。"
        ) from exc


def _response_content(response: Any) -> Any:
    if isinstance(response, (dict, VisualAnalysisResult)):
        return response
    return getattr(response, "content", response)


def _parse_result(response: Any) -> VisualAnalysisResult:
    content = _response_content(response)
    if isinstance(content, VisualAnalysisResult):
        return content
    if isinstance(content, dict):
        return VisualAnalysisResult.model_validate(content)
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    text = str(content).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return VisualAnalysisResult.model_validate(json.loads(text))


class VLMService:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        self.settings = settings
        self._model = model or _create_vlm_model(settings)
        self._structured_model: Any | None = None
        with_structured_output = getattr(self._model, "with_structured_output", None)
        if callable(with_structured_output):
            try:
                self._structured_model = with_structured_output(
                    VisualAnalysisResult,
                    method="json_schema",
                )
            except (TypeError, ValueError, NotImplementedError):
                self._structured_model = None

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str,
        filename: str,
        page_number: int,
    ) -> VisualAnalysisResult:
        data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
        messages = [
            SystemMessage(content=VISUAL_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": build_visual_user_prompt(filename, page_number)},
                    {"type": "image_url", "image_url": data_url},
                ]
            ),
        ]
        model = self._structured_model or self._model
        for attempt in range(2):
            retry_messages = messages
            if attempt:
                retry_messages = [
                    *messages,
                    HumanMessage(
                        content="上一個輸出無法通過 JSON Schema 驗證。請只回傳修正後的 JSON 物件。"
                    ),
                ]
            try:
                return _parse_result(model.invoke(retry_messages))
            except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
                continue
            except Exception as exc:
                raise VLMError("VLM 視覺分析呼叫失敗。") from exc
        raise VLMError("VLM 連續兩次未回傳有效的結構化結果。")


def format_visual_analysis(result: VisualAnalysisResult) -> str:
    sections: list[str] = []

    def add_scalar(title: str, value: str) -> None:
        if value.strip():
            sections.append(f"[{title}]\n{value.strip()}")

    def add_items(title: str, values: list[str]) -> None:
        items = [value.strip() for value in values if value.strip()]
        if items:
            sections.append(f"[{title}]\n" + "\n".join(f"- {item}" for item in items))

    add_scalar("頁面類型", result.page_type)
    add_scalar("視覺摘要", result.summary)
    add_items("可見文字", result.visible_text)
    add_items("可見事實", result.visual_facts)
    add_items("表格內容", result.table_content)
    add_items("圖表觀察", result.chart_insights)
    add_items("可見異常", result.abnormal_visuals)
    add_items("不確定內容", result.uncertain_items)
    return "\n\n".join(sections)


def has_substantive_visual_content(result: VisualAnalysisResult) -> bool:
    """頁面類型或只有不確定內容不足以建立可檢索 Visual Chunk。"""

    return any(
        (
            result.summary.strip(),
            *(item.strip() for item in result.visible_text),
            *(item.strip() for item in result.visual_facts),
            *(item.strip() for item in result.table_content),
            *(item.strip() for item in result.chart_insights),
            *(item.strip() for item in result.abnormal_visuals),
        )
    )
