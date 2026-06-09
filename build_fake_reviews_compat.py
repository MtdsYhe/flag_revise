from __future__ import annotations

import argparse
from pathlib import Path

from fake_reviews_preprocess import build_compatibility_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FLAG-compatible artifacts from the fake reviews CSV.")
    parser.add_argument(
        "--csv",
        default="dataset/final_labeled_fake_reviews_unix.csv",
        help="Path to the labeled fake reviews CSV.",
    )
    parser.add_argument(
        "--output-root",
        default="Amazon",
        help="Directory where Reddit-compatible artifacts will be written.",
    )
    parser.add_argument(
        "--sampler-subdir",
        default="0_10_0",
        help="Subdirectory under output-root for train/val/test sampler files.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the split.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--same-user-k", type=int, default=3, help="Max next-neighbor links per user group.")
    parser.add_argument("--same-product-k", type=int, default=3, help="Max next-neighbor links per product group.")
    parser.add_argument("--batch-size", type=int, default=1, help="NeighborLoader batch size.")
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=[10, 10],
        help="Neighbor sampling fanout, e.g. 10 10.",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used to build node embeddings.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=1200,
        help="Maximum text length sent to the text encoder.",
    )
    args = parser.parse_args()

    result = build_compatibility_dataset(
        args.csv,
        output_root=args.output_root,
        sampler_subdir=args.sampler_subdir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        same_user_k=args.same_user_k,
        same_product_k=args.same_product_k,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        model_name=args.model_name,
        max_text_chars=args.max_text_chars,
    )

    print(f"Wrote graph to {Path(result['data_path'])}")
    print(f"Wrote samplers to {Path(result['train_sampler_path']).parent}")
    print(f"Nodes: {result['num_nodes']}, Edges: {result['num_edges']}")


if __name__ == "__main__":
    main()
