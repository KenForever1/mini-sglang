from __future__ import annotations

import base64
import io
import re
from typing import Union

from PIL import Image


def load_image(source: str) -> Image.Image:
    """Load an image from a URL, base64 data URI, or local file path.

    Args:
        source: One of:
            - http(s):// URL
            - data:image/...;base64,... data URI
            - local file path

    Returns:
        A PIL Image in RGB mode.
    """
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    m = re.match(r"^data:image/[^;]+;base64,(.+)$", source, re.DOTALL)
    if m:
        img_bytes = base64.b64decode(m.group(1))
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return Image.open(source).convert("RGB")
