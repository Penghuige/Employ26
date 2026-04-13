import argparse
import json
from json import JSONDecoder
from pathlib import Path
from typing import Generator, Tuple


INPUT_JSON = Path(r"src\bge\data5\Tier1_Matched_Data.label_studio.json")
OUTPUT_JSON = Path(r"src\bge\data5\Tier1_Matched_Data.label_studio.first20000.json")
DEFAULT_LIMIT = 20000
CHUNK_SIZE = 1024 * 1024


def _iter_json_array_items(path: Path) -> Generator[dict, None, None]:
    """Stream a top-level JSON array from disk without loading the whole file."""
    decoder = JSONDecoder()
    buffer = ""
    started = False
    finished = False

    with path.open("r", encoding="utf-8") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if chunk:
                buffer += chunk
            elif not buffer:
                break

            while True:
                stripped = buffer.lstrip()
                if not stripped:
                    break

                consumed = len(buffer) - len(stripped)
                buffer = stripped

                if not started:
                    if buffer[0] != "[":
                        raise ValueError(f"Top-level JSON is not an array: {path}")
                    started = True
                    buffer = buffer[1:]
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break

                if buffer[0] == "]":
                    finished = True
                    return

                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if chunk:
                        break
                    raise

                yield item
                buffer = buffer[end:].lstrip()

                if buffer.startswith(","):
                    buffer = buffer[1:]
                    continue

                if buffer.startswith("]"):
                    finished = True
                    return

            if not chunk:
                break

    if not finished:
        raise ValueError(f"Incomplete or invalid JSON array: {path}")


def slice_label_studio_json(
    input_json: Path = INPUT_JSON,
    output_json: Path = OUTPUT_JSON,
    limit: int = DEFAULT_LIMIT,
) -> Tuple[int, Path]:
    if limit <= 0:
        raise ValueError("limit must be > 0")
    if not input_json.exists():
        raise FileNotFoundError(f"Input file not found: {input_json}")

    tasks = []
    for idx, item in enumerate(_iter_json_array_items(input_json), start=1):
        tasks.append(item)
        if idx >= limit:
            break

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    return len(tasks), output_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Slice a Label Studio JSON array to the first N tasks.")
    parser.add_argument("--input", type=Path, default=INPUT_JSON, help="Input Label Studio JSON path")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output sliced JSON path")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Number of tasks to keep")
    args = parser.parse_args()

    count, output_path = slice_label_studio_json(
        input_json=args.input,
        output_json=args.output,
        limit=args.limit,
    )
    print(f"[Done] Wrote {count} tasks to: {output_path}")


if __name__ == "__main__":
    main()
