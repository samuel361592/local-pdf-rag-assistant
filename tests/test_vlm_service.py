from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.rag.config import Settings
from src.rag.vlm_service import (
    VLMError,
    VLMService,
    VisualAnalysisResult,
    format_visual_analysis,
)


def valid_payload() -> dict[str, object]:
    return {
        "page_type": "設備照片",
        "summary": "圖片顯示一組齒輪。",
        "visible_text": ["E01"],
        "visual_facts": ["紅色箭頭指向白色齒輪。"],
        "table_content": [],
        "chart_insights": [],
        "abnormal_visuals": ["齒輪邊緣疑似缺損。"],
        "uncertain_items": ["無法確認故障根因。"],
    }


class FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[list[object]] = []

    def invoke(self, messages: list[object]) -> object:
        self.calls.append(messages)
        return self.responses.pop(0)


def test_analyze_image_builds_multimodal_data_url_and_parses_json() -> None:
    model = FakeModel([AIMessage(content=json.dumps(valid_payload()))])
    service = VLMService(Settings(), model=model)

    result = service.analyze_image(
        b"image bytes",
        mime_type="image/png",
        filename="machine.png",
        page_number=1,
    )

    assert result.page_type == "設備照片"
    human = next(message for message in model.calls[0] if isinstance(message, HumanMessage))
    assert human.content[1]["image_url"].startswith("data:image/png;base64,")
    assert "machine.png" in human.content[0]["text"]


def test_invalid_json_retries_only_once_then_raises_safe_error() -> None:
    model = FakeModel([AIMessage(content="invalid"), AIMessage(content="still invalid")])
    service = VLMService(Settings(), model=model)

    with pytest.raises(VLMError) as raised:
        service.analyze_image(
            b"secret image bytes",
            mime_type="image/jpeg",
            filename="secret.jpg",
            page_number=2,
        )

    assert len(model.calls) == 2
    assert "c2VjcmV0" not in str(raised.value)
    assert "Data URL" not in str(raised.value)


def test_format_visual_analysis_omits_empty_sections() -> None:
    text = format_visual_analysis(
        VisualAnalysisResult(
            page_type="圖表",
            summary="營收呈上升趨勢。",
            chart_insights=["最高點位於第四季。"],
        )
    )

    assert "[頁面類型]" in text
    assert "[圖表觀察]" in text
    assert "[表格內容]" not in text
    assert "[不確定內容]" not in text


def test_settings_validate_vlm_options() -> None:
    with pytest.raises(ValueError, match="VLM_MODE"):
        Settings(vlm_mode="sometimes")
    with pytest.raises(ValueError, match="VLM_MODEL"):
        Settings(vlm_enabled=True, vlm_model="")
    with pytest.raises(ValueError, match="VLM_MAX_IMAGE_EDGE"):
        Settings(vlm_max_image_edge=0)
