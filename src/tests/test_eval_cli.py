"""测试 eval_cli 命令。"""
import json
import pytest
from pathlib import Path
from src.skill_extraction.eval_cli import build_parser, cmd_list


SAMPLE_RECORD = {
    "dict_version": "v1",
    "evaluated_at": "2026-06-12T14:00:00",
    "soft_skill_metrics": {
        "coverage": 0.1141,
        "precision": 0.0876,
        "dimension_accuracy": 0.8495,
    },
    "hard_skill_metrics": {
        "precision": 0.7018,
        "recall": 0.9053,
        "f1": 0.7907,
        "category_accuracy": 1.0,
    },
    "gold_source": "annotations.label_studio_tasks_v2",
    "sample_count": 300,
}


class TestBuildParser:
    def test_list_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["list"])
        assert args.command == "list"

    def test_run_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["run"])
        assert args.command == "run"

    def test_compare_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["compare", "v1", "v2"])
        assert args.command == "compare"
        assert args.version_a == "v1"
        assert args.version_b == "v2"

    def test_no_command_shows_help(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])


class TestCmdList:
    def test_empty_registry(self, tmp_path, capsys):
        from src.skill_extraction._eval_registry import load_registry
        load_registry(tmp_path)  # create empty registry
        cmd_list(tmp_path)
        captured = capsys.readouterr()
        assert "no eval" in captured.out.lower() or "评估" in captured.out


class TestCmdRun:
    def test_run_creates_registry_record(self, tmp_path, monkeypatch):
        import json
        from src.skill_extraction.eval_cli import cmd_run

        gold_dir = tmp_path / "gold"
        gold_dir.mkdir()
        hard_data = [
            {
                "sample_id": "test_1",
                "text": "use Java and MySQL",
                "gold_skills": ["Java", "MySQL"],
                "gold_categories": {"Java": "programming_language"},
            }
        ]
        soft_data = [
            {
                "sample_id": "test_1",
                "text": "have communication skills and responsibility",
                "gold_soft_skills": [
                    {"name": "communication", "dimension": "extraversion"},
                    {"name": "responsibility", "dimension": "conscientiousness"},
                ],
            }
        ]
        (gold_dir / "hard.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in hard_data)
        )
        (gold_dir / "soft.jsonl").write_text(
            "\n".join(json.dumps(d, ensure_ascii=False) for d in soft_data)
        )

        eval_dir = tmp_path / "eval"
        cmd_run(
            eval_dir=eval_dir,
            hard_dataset=gold_dir / "hard.jsonl",
            soft_dataset=gold_dir / "soft.jsonl",
        )

        with open(eval_dir / "registry.json", "r", encoding="utf-8") as f:
            registry = json.load(f)
        assert len(registry["evaluations"]) == 1
        record = registry["evaluations"][0]
        assert record["dict_version"] == "v1"
        assert "soft_skill_metrics" in record
        assert "hard_skill_metrics" in record

        version_dir = eval_dir / "v1"
        assert version_dir.exists()
        assert (version_dir / "summary.json").exists()
