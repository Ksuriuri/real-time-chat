from __future__ import annotations

import argparse
import time
from statistics import mean, median

from ark_language_model import ArkLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Ark language-model TTFT.")
    parser.add_argument(
        "--model",
        default=None,
        help="Ark model id. Defaults to ARK_MODEL or doubao-seed-2-0-mini-260428.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=15)
    parser.add_argument(
        "--prompt",
        default="请用一句话解释：为什么低延迟对实时语音对话很重要？",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Leave Ark thinking mode enabled. Default disables thinking for lower latency.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = ArkLanguageModel(model=args.model)
    rows: list[tuple[float, float]] = []

    for run in range(1, args.runs + 1):
        start_time = time.perf_counter()
        first_token_s: float | None = None
        text_parts: list[str] = []

        for token in llm.stream_text(
            args.prompt,
            max_output_tokens=args.max_output_tokens,
            thinking_disabled=not args.enable_thinking,
        ):
            if first_token_s is None:
                first_token_s = token.elapsed_s
            text_parts.append(token.text)

        done_s = time.perf_counter() - start_time
        rows.append((first_token_s or done_s, done_s))
        print(
            f"run={run} first_token_ms={rows[-1][0] * 1000:.1f} "
            f"done_ms={done_s * 1000:.1f} text={''.join(text_parts)!r}",
            flush=True,
        )

    first_tokens = [row[0] for row in rows]
    done_times = [row[1] for row in rows]
    print(
        "summary "
        f"first_token_ms avg={mean(first_tokens) * 1000:.1f} "
        f"median={median(first_tokens) * 1000:.1f} "
        f"done_ms avg={mean(done_times) * 1000:.1f} "
        f"median={median(done_times) * 1000:.1f}"
    )


if __name__ == "__main__":
    main()
