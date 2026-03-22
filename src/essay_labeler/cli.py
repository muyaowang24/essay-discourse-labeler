from __future__ import annotations

import argparse

from essay_labeler.config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="essay-labeler")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train discourse labeling models.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--set", action="append", default=[])

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a checkpoint on a validation split.")
    eval_parser.add_argument("--config", required=True)
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--set", action="append", default=[])

    predict_parser = subparsers.add_parser("predict", help="Generate prediction files.")
    predict_parser.add_argument("--config", required=True)
    predict_parser.add_argument("--checkpoint", required=True, action="append")
    predict_parser.add_argument("--input-dir", required=True)
    predict_parser.add_argument("--output", required=True)
    predict_parser.add_argument("--set", action="append", default=[])
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, overrides=args.set)

    if args.command == "train":
        from essay_labeler.train import train

        metrics = train(config)
        for row in metrics:
            print(f"fold={row['fold']} macro_f1={row['macro_f1']:.6f} checkpoint={row['checkpoint']}")
    elif args.command == "evaluate":
        from essay_labeler.train import evaluate

        results = evaluate(config, checkpoint=args.checkpoint)
        print(f"macro_f1={results['macro_f1']:.6f}")
        print(f"predictions_path={results['predictions_path']}")
    elif args.command == "predict":
        from essay_labeler.train import predict

        output = predict(
            config,
            checkpoints=args.checkpoint,
            input_dir=args.input_dir,
            output=args.output,
        )
        print(output)


if __name__ == "__main__":
    main()
