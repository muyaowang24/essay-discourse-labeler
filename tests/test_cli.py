from essay_labeler.cli import build_parser


def test_cli_predict_accepts_multiple_checkpoints() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--config",
            "configs/base.yaml",
            "--checkpoint",
            "a.pt",
            "--checkpoint",
            "b.pt",
            "--input-dir",
            "data/test",
            "--output",
            "artifacts/out.csv",
        ]
    )
    assert args.checkpoint == ["a.pt", "b.pt"]
