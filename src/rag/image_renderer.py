"""PDF page rendering and uploaded-image normalization utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps


class ImageRenderError(RuntimeError):
    """Raised when a document page or image cannot be rendered safely."""


def render_pdf_page_to_image(
    pdf_path: Path,
    page_index: int,
    *,
    dpi: int,
) -> bytes:
    """將指定 PDF 頁面渲染成 PNG bytes。"""

    try:
        import pymupdf
    except ImportError as exc:
        raise ImageRenderError("找不到 PyMuPDF，無法將 PDF 頁面轉成圖片。") from exc

    try:
        with pymupdf.open(pdf_path) as pdf:
            page = pdf.load_page(page_index)
            return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")
    except Exception as exc:
        raise ImageRenderError(f"無法渲染 PDF 第 {page_index + 1} 頁。") from exc


def pdf_page_has_images(pdf_path: Path, page_index: int) -> bool:
    """回傳 PDF 頁面是否包含至少一張內嵌點陣圖片。"""

    try:
        import pymupdf
    except ImportError as exc:
        raise ImageRenderError("找不到 PyMuPDF，無法檢查 PDF 頁面圖片。") from exc

    try:
        with pymupdf.open(pdf_path) as pdf:
            return bool(pdf.load_page(page_index).get_images(full=True))
    except Exception as exc:
        raise ImageRenderError(f"無法檢查 PDF 第 {page_index + 1} 頁的圖片。") from exc


def normalize_image(
    image_bytes: bytes,
    *,
    max_edge: int,
) -> tuple[bytes, str]:
    """修正方向、轉換色彩並等比例縮圖，回傳圖片 bytes 與 MIME type。"""

    if max_edge <= 0:
        raise ValueError("max_edge 必須大於 0。")
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image = ImageOps.exif_transpose(source)
            image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            has_alpha = image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            )
            output = BytesIO()
            if has_alpha:
                image.convert("RGBA").save(output, format="PNG", optimize=True)
                mime_type = "image/png"
            else:
                image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
                mime_type = "image/jpeg"
            return output.getvalue(), mime_type
    except Exception as exc:
        raise ImageRenderError("無法讀取或正規化圖片。") from exc
