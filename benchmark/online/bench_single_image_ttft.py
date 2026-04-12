from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import List

from minisgl.benchmark.client import get_model_name
from minisgl.utils import init_logger
from openai import AsyncOpenAI as OpenAI

logger = init_logger(__name__)


@dataclass(frozen=True)
class RunResult:
    ttft_ms: float
    e2e_ms: float
    output_chars: int


def _percentile(values: List[float], ratio: float) -> float:
    assert values, "values must not be empty"
    idx = min(len(values) - 1, max(0, int(len(values) * ratio)))
    return sorted(values)[idx]


def _build_messages(prompt: str, image_source: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_source}},
            ],
        }
    ]


async def bench_one(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    image_source: str,
    max_tokens: int,
) -> RunResult:
    start = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        stream=True,
        messages=_build_messages(prompt, image_source),
        max_tokens=max_tokens,
        temperature=0.0,
    )

    first_stream_at = None
    first_token_at = None
    output_parts: List[str] = []
    async for chunk in response:
        now = time.perf_counter()
        if first_stream_at is None:
            first_stream_at = now

        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = delta.content or ""
        if content:
            output_parts.append(content)
            if first_token_at is None:
                first_token_at = now

    end = time.perf_counter()
    ttft_anchor = first_token_at or first_stream_at or end
    return RunResult(
        ttft_ms=(ttft_anchor - start) * 1000,
        e2e_ms=(end - start) * 1000,
        output_chars=sum(len(part) for part in output_parts),
    )


def summarize(results: List[RunResult], label: str) -> None:
    ttfts = [item.ttft_ms for item in results]
    e2es = [item.e2e_ms for item in results]
    logger.info(
        "%s runs=%d | TTFT avg=%.2f ms p50=%.2f ms p90=%.2f ms max=%.2f ms | E2E avg=%.2f ms",
        label,
        len(results),
        statistics.mean(ttfts),
        _percentile(ttfts, 0.5),
        _percentile(ttfts, 0.9),
        max(ttfts),
        statistics.mean(e2es),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TTFT for a single-image chat request.")
    parser.add_argument("--image", required=True, help="Local path, data URI, or URL for the image.")
    parser.add_argument("--prompt", default="请用一句话描述这张图片。", help="Prompt to pair with the image.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum output tokens.")
    parser.add_argument("--runs", type=int, default=5, help="Measured runs after warmup.")
    parser.add_argument("--warmup-runs", type=int, default=1, help="Warmup runs before measurement.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=1919, help="Server port.")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}/v1"
    async with OpenAI(base_url=base_url, api_key="") as client:
        model = await get_model_name(client)
        logger.info("Benchmarking model %s via %s", model, base_url)

        warmup_results: List[RunResult] = []
        for idx in range(args.warmup_runs):
            result = await bench_one(
                client,
                model=model,
                prompt=args.prompt,
                image_source=args.image,
                max_tokens=args.max_tokens,
            )
            warmup_results.append(result)
            logger.info(
                "Warmup %d/%d | TTFT=%.2f ms | E2E=%.2f ms | output_chars=%d",
                idx + 1,
                args.warmup_runs,
                result.ttft_ms,
                result.e2e_ms,
                result.output_chars,
            )

        measured_results: List[RunResult] = []
        for idx in range(args.runs):
            result = await bench_one(
                client,
                model=model,
                prompt=args.prompt,
                image_source=args.image,
                max_tokens=args.max_tokens,
            )
            measured_results.append(result)
            logger.info(
                "Run %d/%d | TTFT=%.2f ms | E2E=%.2f ms | output_chars=%d",
                idx + 1,
                args.runs,
                result.ttft_ms,
                result.e2e_ms,
                result.output_chars,
            )

    if warmup_results:
        summarize(warmup_results, "Warmup")
    summarize(measured_results, "Measured")


if __name__ == "__main__":
    asyncio.run(main())
