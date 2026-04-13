import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List


DEFAULT_INPUT = r"src\bge\data5\Tier1_Matched_Data.label_studio_shards"
DEFAULT_OUTPUT = r"src\bge\data5\Tier1_Matched_Data.label_studio_shards_formatted"


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def format_job_requirements(text: str) -> str:
    text = normalize_newlines(text or "").strip()
    if not text:
        return ""

    # Normalize common inline separators first.
    text = re.sub(r"\s*\|\s*", " |\n", text)
    text = re.sub(r"\s*[;；]\s*", "；\n", text)

    # Add line breaks before common numbered items when they appear inline.
    patterns = [
        r"(?<!\n)(?=(?:\d{1,2}|[一二三四五六七八九十]+)[、.．])",
        r"(?<!\n)(?=(?:\d{1,2})[)\）])",
        r"(?<!\n)(?=(?:任职要求|岗位职责|工作职责|职位要求|职位描述|福利待遇|任职资格)\s*[:：])",
        r"(?<!\n)(?=(?:-|\*)\s*)",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "\n", text)

    # Collapse excessive blank lines and trim line-wise.
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def iter_json_files(input_path: Path) -> Iterable[Path]:
    if input_path.is_file():
        yield input_path
        return

    for path in sorted(input_path.glob("*.json")):
        if path.is_file():
            yield path


def process_json_file(input_file: Path, output_file: Path) -> int:
    with input_file.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    count = 0
    for task in tasks:
        data = task.get("data", {})
        if "job_requirements_clean" in data:
            data["job_requirements_clean"] = format_job_requirements(data.get("job_requirements_clean", ""))
            count += 1

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    return count


def build_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / input_file.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Format job_requirements_clean for Label Studio JSON files.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON file or directory.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSON file or directory.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    processed_files: List[str] = []
    total_rows = 0
    for json_file in iter_json_files(input_path):
        target_path = build_output_path(json_file, input_path, output_path)
        row_count = process_json_file(json_file, target_path)
        processed_files.append(str(target_path))
        total_rows += row_count

    print(f"[Done] Processed files: {len(processed_files)}")
    print(f"[Done] Updated job_requirements_clean rows: {total_rows}")
    for path in processed_files[:10]:
        print(f"  - {path}")
    if len(processed_files) > 10:
        print(f"  ... and {len(processed_files) - 10} more")


if __name__ == "__main__":
    main()
