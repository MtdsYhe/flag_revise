import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fake_reviews_preprocess import (
    clean_review_text,
    build_review_edges,
    stratified_split_indices,
    build_compatibility_dataset,
)


class FakeReviewPreprocessTests(unittest.TestCase):
    def test_clean_review_text_joins_title_and_text(self):
        text = clean_review_text("  Great Product!  ", "Loved it<br />Works well &amp; fast")
        self.assertEqual(text, "Great Product! Loved it Works well & fast")

    def test_build_review_edges_connects_same_user_and_product(self):
        rows = [
            {
                "user_id": "u1",
                "parent_asin": "p1",
                "asin": "a1",
                "unix_timestamp": 10,
            },
            {
                "user_id": "u1",
                "parent_asin": "p2",
                "asin": "a2",
                "unix_timestamp": 20,
            },
            {
                "user_id": "u2",
                "parent_asin": "p1",
                "asin": "a1",
                "unix_timestamp": 15,
            },
        ]

        edge_index = build_review_edges(rows, same_user_k=1, same_product_k=1)
        actual = {tuple(pair) for pair in edge_index.t().tolist()}

        self.assertIn((0, 1), actual)
        self.assertIn((1, 0), actual)
        self.assertIn((0, 2), actual)
        self.assertIn((2, 0), actual)

    def test_stratified_split_indices_preserve_all_items_once(self):
        labels = [0, 0, 1, 1, 1, 0]
        train_idx, val_idx, test_idx = stratified_split_indices(
            labels,
            train_ratio=0.5,
            val_ratio=0.25,
            seed=7,
        )

        all_idx = train_idx + val_idx + test_idx
        self.assertEqual(sorted(all_idx), list(range(len(labels))))
        self.assertEqual(len(set(all_idx)), len(labels))
        self.assertEqual(len(train_idx), 3)
        self.assertEqual(len(val_idx), 1)
        self.assertEqual(len(test_idx), 2)

    def test_build_compatibility_dataset_creates_reddit_artifacts(self):
        try:
            import csv
            import torch
        except Exception:
            self.skipTest("runtime dependencies not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            csv_path = tmpdir / "mini.csv"
            out_dir = tmpdir / "out"

            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "rating",
                        "title",
                        "text",
                        "images",
                        "asin",
                        "parent_asin",
                        "user_id",
                        "timestamp",
                        "helpful_vote",
                        "verified_purchase",
                        "label",
                        "user_timestamp",
                        "user_review_burst",
                        "unix_timestamp",
                    ]
                )
                for i in range(10):
                    writer.writerow(
                        [
                            5 if i % 2 == 0 else 1,
                            f"Title {i}",
                            f"Review body {i}",
                            "[]",
                            f"A{i // 2}",
                            f"P{i // 2}",
                            f"U{i}",
                            f"1/{i + 1}/2021 0:00",
                            0,
                            "TRUE",
                            i % 2,
                            99999,
                            99999,
                            1600000000 + i,
                        ]
                    )

            with patch(
                "fake_reviews_preprocess._encode_texts",
                return_value=torch.zeros((10, 384), dtype=torch.float32),
            ):
                result = build_compatibility_dataset(
                    csv_path,
                    output_root=out_dir,
                    sampler_subdir="0_10_0",
                    same_user_k=1,
                    same_product_k=1,
                    batch_size=1,
                    num_neighbors=(1,),
                    max_text_chars=50,
                    seed=7,
                )

            self.assertTrue((out_dir / "amazon.pt").exists())
            self.assertTrue((out_dir / "0_10_0" / "train_sampler2.pt").exists())
            self.assertTrue((out_dir / "0_10_0" / "val_sampler2.pt").exists())
            self.assertTrue((out_dir / "0_10_0" / "test_sampler2.pt").exists())
            self.assertEqual(result["num_nodes"], 10)


if __name__ == "__main__":
    unittest.main()
