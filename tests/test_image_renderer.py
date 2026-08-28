from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.rag.image_renderer import normalize_image


def encode_image(size: tuple[int, int], mode: str = "RGB") -> bytes:
    output = BytesIO()
    color = (255, 255, 255, 0) if mode == "RGBA" else "white"
    Image.new(mode, size, color).save(output, format="PNG")
    return output.getvalue()


def test_normalize_image_resizes_without_changing_aspect_ratio() -> None:
    normalized, mime_type = normalize_image(encode_image((200, 100)), max_edge=80)

    with Image.open(BytesIO(normalized)) as image:
        assert image.size == (80, 40)
        assert image.mode == "RGB"
    assert mime_type == "image/jpeg"


def test_normalize_image_does_not_upscale_and_preserves_alpha() -> None:
    normalized, mime_type = normalize_image(
        encode_image((20, 10), mode="RGBA"),
        max_edge=80,
    )

    with Image.open(BytesIO(normalized)) as image:
        assert image.size == (20, 10)
        assert image.mode == "RGBA"
    assert mime_type == "image/png"
