from __future__ import annotations

import base64
import io
import re
import time

from PIL import Image
from minisgl.utils import init_logger, maybe_log_perf

logger = init_logger(__name__)


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
    start_time = time.perf_counter()
    source_kind = "local_path"

    if source.startswith(("http://", "https://")):
        import requests

        source_kind = "remote_url"
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", source, re.DOTALL)
        if m:
            source_kind = "data_uri"
            img_bytes = base64.b64decode(m.group(1))
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            image = Image.open(source).convert("RGB")

    maybe_log_perf(logger, f"image_io.load_image source={source_kind}", start_time)
    return image
