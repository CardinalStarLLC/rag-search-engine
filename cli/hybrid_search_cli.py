import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize = subparsers.add_parser("normalize", help="Normalize text for hybrid search")
    normalize.add_argument("scores", type=float, nargs='+', help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            scores = args.scores
            if not scores:
                return
            min_score = min(scores)
            max_score = max(scores)
            if min_score == max_score:
                for _ in scores:
                    print(1.0)
            else:
                for score in scores:
                    normalized = (score - min_score) / (max_score - min_score)
                    print(f"* {normalized:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()