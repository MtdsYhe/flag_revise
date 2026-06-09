#!/usr/bin/env python3
"""
Remove duplicate user_ids from the Amazon fake reviews dataset.
Keeps only the first review per user (chronologically by unix_timestamp).
"""

import pandas as pd
from pathlib import Path


def deduplicate_users(input_csv: str, output_csv: str):
    """
    Load CSV, remove duplicate user_ids, save deduplicated version.

    Args:
        input_csv: Path to original CSV with potential duplicate users
        output_csv: Path to save deduplicated CSV
    """
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv)

    original_count = len(df)
    print(f"Original dataset: {original_count} reviews")

    unique_users = df['user_id'].nunique()
    print(f"Unique users: {unique_users}")

    # Sort by user_id and unix_timestamp to ensure we keep the earliest review per user
    df = df.sort_values(['user_id', 'unix_timestamp'])

    # Keep first review per user
    df_dedup = df.drop_duplicates(subset='user_id', keep='first')

    dedup_count = len(df_dedup)
    removed_count = original_count - dedup_count

    print(f"After deduplication: {dedup_count} reviews")
    print(f"Removed: {removed_count} reviews ({removed_count/original_count*100:.2f}%)")

    # Check label distribution
    print("\nLabel distribution in deduplicated dataset:")
    print(df_dedup['label'].value_counts().sort_index())

    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_dedup.to_csv(output_csv, index=False)
    print(f"\nSaved to: {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Deduplicate users in Amazon fake reviews dataset')
    parser.add_argument(
        '--input',
        default='dataset/final_labeled_fake_reviews_unix.csv',
        help='Input CSV path'
    )
    parser.add_argument(
        '--output',
        default='dataset/final_labeled_fake_reviews_unix_dedup.csv',
        help='Output CSV path'
    )

    args = parser.parse_args()
    deduplicate_users(args.input, args.output)
