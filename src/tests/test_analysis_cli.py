from src.analysis import cli


def test_build_parser_accepts_structured_defaults():
    args = cli.build_parser().parse_args(["structured"])

    assert args.command == "structured"
    assert args.structured_command is None
    assert args.with_integration is False
    assert args.with_excel is False
    assert args.skip_standardized is False


def test_build_parser_accepts_structured_run_defaults():
    args = cli.build_parser().parse_args(["structured", "run"])

    assert args.command == "structured"
    assert args.structured_command == "run"
    assert args.with_integration is False


def test_build_parser_accepts_structured_options():
    args = cli.build_parser().parse_args(
        ["structured", "--with-integration", "--sample", "--with-excel"]
    )

    assert args.command == "structured"
    assert args.with_integration is True
    assert args.sample is True
    assert args.with_excel is True


def test_build_parser_accepts_structured_run_options():
    args = cli.build_parser().parse_args(
        ["structured", "run", "--with-integration", "--sample", "--with-excel"]
    )

    assert args.command == "structured"
    assert args.structured_command == "run"
    assert args.with_integration is True
    assert args.sample is True
    assert args.with_excel is True


def test_build_parser_accepts_requirements_options():
    args = cli.build_parser().parse_args(
        ["requirements", "--top-n", "5", "--min-group-size", "2"]
    )

    assert args.command == "requirements"
    assert args.top_n == 5
    assert args.min_group_size == 2


def test_build_parser_accepts_requirements_run_options():
    args = cli.build_parser().parse_args(
        ["requirements", "run", "--top-n", "5", "--min-group-size", "2"]
    )

    assert args.command == "requirements"
    assert args.requirements_command == "run"
    assert args.top_n == 5
    assert args.min_group_size == 2
