from __future__ import annotations

import csv
import html
import math
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is optional for import-time tests
    torch = None

try:
    from torch_geometric.data import Data  # type: ignore
except Exception:  # pragma: no cover - torch_geometric is optional for import-time tests
    Data = None


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SimpleEdgeIndex:
    """A light fallback that behaves enough like a tensor for tests."""

    edges: tuple[tuple[int, int], ...]

    def t(self) -> "SimpleEdgeIndex":
        return self

    def tolist(self) -> list[list[int]]:
        return [list(edge) for edge in self.edges]

    def __len__(self) -> int:
        return len(self.edges)


@dataclass(frozen=True)
class ReviewRecord:
    node_id: int
    user_id: str
    asin: str
    parent_asin: str
    unix_timestamp: int
    label: int
    raw_text: str
    rating: float | None = None
    helpful_vote: int | None = None
    verified_purchase: str | None = None


@dataclass
class SimpleNeighborBatch:
    subset: Any
    central: int
    edge_index: Any
    n_id: Any


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _clean_piece(value: Any) -> str:
    text = html.unescape(_as_text(value))
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_review_text(title: Any, text: Any) -> str:
    title_text = _clean_piece(title)
    body_text = _clean_piece(text)
    if title_text and body_text:
        return f"{title_text} {body_text}"
    return title_text or body_text


def load_review_rows(csv_path: str | os.PathLike[str]) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_review_records(csv_path: str | os.PathLike[str]) -> list[ReviewRecord]:
    rows = load_review_rows(csv_path)
    records: list[ReviewRecord] = []
    for node_id, row in enumerate(rows):
        records.append(
            ReviewRecord(
                node_id=node_id,
                user_id=_as_text(row.get("user_id")).strip(),
                asin=_as_text(row.get("asin")).strip(),
                parent_asin=_as_text(row.get("parent_asin")).strip(),
                unix_timestamp=_parse_int(row.get("unix_timestamp") or row.get("timestamp")),
                label=_parse_int(row.get("label")),
                raw_text=clean_review_text(row.get("title"), row.get("text")),
                rating=_parse_float(row.get("rating")),
                helpful_vote=_parse_int(row.get("helpful_vote"), default=0),
                verified_purchase=_as_text(row.get("verified_purchase")).strip() or None,
            )
        )
    return records


def _group_key_for_product(record: ReviewRecord) -> str | None:
    if record.parent_asin:
        return record.parent_asin
    if record.asin:
        return record.asin
    return None


def _sorted_group_indices(records: Sequence[ReviewRecord], key_fn) -> list[list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        key = key_fn(record)
        if key is None:
            continue
        groups[str(key)].append(idx)

    ordered_groups: list[list[int]] = []
    for indices in groups.values():
        indices.sort(key=lambda i: (records[i].unix_timestamp, i))
        ordered_groups.append(indices)
    return ordered_groups


def build_review_edges(
    rows: Sequence[Mapping[str, Any]] | Sequence[ReviewRecord],
    *,
    same_user_k: int = 3,
    same_product_k: int = 3,
) -> SimpleEdgeIndex:
    """Build sparse undirected review-review edges from shared user/product groups.

    The graph is intentionally sparse: each node is connected to the next `k`
    reviews inside the same user/product group after chronological sorting.
    """

    records: list[ReviewRecord] = []
    for idx, row in enumerate(rows):
        if isinstance(row, ReviewRecord):
            records.append(row)
            continue
        records.append(
            ReviewRecord(
                node_id=idx,
                user_id=_as_text(row.get("user_id")).strip(),
                asin=_as_text(row.get("asin")).strip(),
                parent_asin=_as_text(row.get("parent_asin")).strip(),
                unix_timestamp=_parse_int(row.get("unix_timestamp") or row.get("timestamp")),
                label=_parse_int(row.get("label")),
                raw_text=clean_review_text(row.get("title"), row.get("text")),
                rating=_parse_float(row.get("rating")),
                helpful_vote=_parse_int(row.get("helpful_vote"), default=0),
                verified_purchase=_as_text(row.get("verified_purchase")).strip() or None,
            )
        )

    edge_set: set[tuple[int, int]] = set()

    def add_group_edges(groups: Iterable[list[int]], k: int) -> None:
        if k <= 0:
            return
        for indices in groups:
            n = len(indices)
            if n < 2:
                continue
            for pos, src in enumerate(indices):
                upper = min(n, pos + k + 1)
                for dst_pos in range(pos + 1, upper):
                    dst = indices[dst_pos]
                    if src == dst:
                        continue
                    edge_set.add((src, dst))
                    edge_set.add((dst, src))

    add_group_edges(_sorted_group_indices(records, lambda r: r.user_id or None), same_user_k)
    add_group_edges(_sorted_group_indices(records, _group_key_for_product), same_product_k)

    ordered_edges = tuple(sorted(edge_set))
    return SimpleEdgeIndex(ordered_edges)


def stratified_split_indices(
    labels: Sequence[int],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    n = len(labels)
    if n == 0:
        return [], [], []

    test_ratio = 1.0 - train_ratio - val_ratio
    target_counts = [
        int(n * train_ratio),
        int(n * val_ratio),
        n - int(n * train_ratio) - int(n * val_ratio),
    ]
    ratios = [train_ratio, val_ratio, test_ratio]
    split_names = ["train", "val", "test"]

    grouped: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped[int(label)].append(idx)

    rng = random.Random(seed)
    per_split: dict[str, list[int]] = {name: [] for name in split_names}
    global_remaining = target_counts[:]

    for label in sorted(grouped):
        indices = grouped[label]
        rng.shuffle(indices)
        group_size = len(indices)
        if group_size == 0:
            continue

        raw_quota = [group_size * ratio for ratio in ratios]
        allocated = [min(int(math.floor(value)), group_size) for value in raw_quota]
        allocated_total = sum(allocated)
        while allocated_total > group_size:
            # Defensive normalization when rounding oddities happen.
            for i in range(2, -1, -1):
                if allocated_total <= group_size:
                    break
                if allocated[i] > 0:
                    allocated[i] -= 1
                    allocated_total -= 1

        for split_idx, amount in enumerate(allocated):
            take = min(amount, global_remaining[split_idx])
            if take:
                per_split[split_names[split_idx]].extend(indices[:take])
                indices = indices[take:]
                allocated[split_idx] -= take
                global_remaining[split_idx] -= take

        remainders = [raw_quota[i] - allocated[i] for i in range(3)]
        while indices:
            # Pick the split with the highest remaining fractional demand and capacity.
            best_split = None
            best_score = None
            for split_idx in range(3):
                if global_remaining[split_idx] <= 0:
                    continue
                score = (remainders[split_idx], -split_idx)
                if best_score is None or score > best_score:
                    best_score = score
                    best_split = split_idx
            if best_split is None:
                best_split = next(i for i in range(3) if global_remaining[i] > 0)
            per_split[split_names[best_split]].append(indices.pop(0))
            global_remaining[best_split] -= 1
            remainders[best_split] = 0.0

    # If any global capacity still remains due to corner cases, fill in order.
    leftover = [idx for name in split_names for idx in per_split[name]]
    assigned = set(leftover)
    if len(assigned) != n:
        missing = [i for i in range(n) if i not in assigned]
        cursor = 0
        for split_idx, remaining in enumerate(global_remaining):
            if remaining <= 0:
                continue
            take = missing[cursor : cursor + remaining]
            per_split[split_names[split_idx]].extend(take)
            cursor += remaining

    return per_split["train"], per_split["val"], per_split["test"]


def _edge_index_to_tensor(edges: Sequence[tuple[int, int]]):
    if torch is None:
        return SimpleEdgeIndex(tuple(edges))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _indices_to_mask(indices: Sequence[int], size: int):
    if torch is None:
        mask = [False] * size
        for idx in indices:
            mask[idx] = True
        return mask
    mask = torch.zeros(size, dtype=torch.bool)
    if indices:
        mask[torch.tensor(indices, dtype=torch.long)] = True
    return mask


def _encode_texts(texts: Sequence[str], model_name: str, max_chars: int):
    if torch is None:
        raise RuntimeError("torch is required to build embeddings")
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise RuntimeError("sentence-transformers is required to build embeddings") from exc

    encoder = SentenceTransformer(model_name)
    clipped = [text[:max_chars] if max_chars and len(text) > max_chars else text for text in texts]
    embeddings = encoder.encode(clipped)
    return torch.tensor(embeddings, dtype=torch.float32)


def _edge_index_pairs(edge_index) -> list[tuple[int, int]]:
    if torch is not None and isinstance(edge_index, torch.Tensor):
        matrix = edge_index.detach().cpu().tolist()
        if not matrix:
            return []
        if len(matrix) == 2:
            return list(zip(matrix[0], matrix[1]))
        return [tuple(pair) for pair in matrix]
    if hasattr(edge_index, "edges"):
        return [tuple(edge) for edge in edge_index.edges]
    if hasattr(edge_index, "tolist"):
        raw = edge_index.tolist()
        if not raw:
            return []
        if len(raw) == 2 and all(isinstance(row, list) for row in raw):
            return list(zip(raw[0], raw[1]))
        return [tuple(pair) for pair in raw]
    raise TypeError(f"Unsupported edge_index type: {type(edge_index)!r}")


def _build_adjacency(edge_index, num_nodes: int) -> list[list[int]]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for src, dst in _edge_index_pairs(edge_index):
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            adjacency[src].append(dst)
    return [sorted(set(neighbors)) for neighbors in adjacency]


def _sample_nodes_for_seed(seed: int, adjacency: Sequence[Sequence[int]], num_neighbors: Sequence[int]) -> list[int]:
    selected = [seed]
    seen = {seed}
    frontier = [seed]

    if not num_neighbors:
        return selected

    for fanout in num_neighbors:
        if not frontier:
            break
        next_frontier: list[int] = []
        for node in frontier:
            candidates = [neighbor for neighbor in adjacency[node] if neighbor not in seen]
            if fanout is not None and fanout >= 0:
                candidates = candidates[: int(fanout)]
            for neighbor in candidates:
                seen.add(neighbor)
                selected.append(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier

    return selected


def _build_local_edge_index(edge_pairs: Sequence[tuple[int, int]], selected: Sequence[int]):
    local_pos = {node: idx for idx, node in enumerate(selected)}
    selected_set = set(selected)
    local_pairs = [
        (local_pos[src], local_pos[dst])
        for src, dst in edge_pairs
        if src in selected_set and dst in selected_set
    ]
    if torch is None:
        return SimpleEdgeIndex(tuple(local_pairs))
    if not local_pairs:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(local_pairs, dtype=torch.long).t().contiguous()


def _materialize_neighbor_batches(
    data,
    input_nodes,
    *,
    num_neighbors: Sequence[int],
    batch_size: int,
):
    if torch is None:
        raise RuntimeError("torch is required to build sampler batches")
    if batch_size != 1:
        raise ValueError("This compatibility sampler currently supports batch_size=1 only")

    if isinstance(input_nodes, torch.Tensor):
        seed_nodes = [int(node) for node in input_nodes.detach().cpu().tolist()]
    else:
        seed_nodes = [int(node) for node in input_nodes]

    num_nodes = int(getattr(data, "num_nodes", 0) or data.x.size(0))
    adjacency = _build_adjacency(data.edge_index, num_nodes)
    edge_pairs = _edge_index_pairs(data.edge_index)

    batches: list[SimpleNeighborBatch] = []
    for seed in seed_nodes:
        selected = _sample_nodes_for_seed(seed, adjacency, list(num_neighbors))
        subset = torch.tensor(selected, dtype=torch.long)
        edge_index = _build_local_edge_index(edge_pairs, selected)
        batches.append(
            SimpleNeighborBatch(
                subset=subset,
                central=seed,
                edge_index=edge_index,
                n_id=subset,
            )
        )

    return batches


def build_compatibility_dataset(
    csv_path: str | os.PathLike[str],
    *,
    output_root: str | os.PathLike[str] = "Amazon",
    sampler_subdir: str = "0_10_0",
    model_name: str = "all-MiniLM-L6-v2",
    max_text_chars: int = 1200,
    same_user_k: int = 3,
    same_product_k: int = 3,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
    batch_size: int = 1,
    num_neighbors: Sequence[int] = (10, 10),
):
    if torch is None or Data is None:
        raise RuntimeError("torch and torch_geometric are required to build the dataset")

    records = load_review_records(csv_path)
    texts = [record.raw_text for record in records]
    labels = [record.label for record in records]
    user_ids = [record.user_id for record in records]
    asin = [record.asin for record in records]
    parent_asin = [record.parent_asin for record in records]
    timestamps = [record.unix_timestamp for record in records]

    train_idx, val_idx, test_idx = stratified_split_indices(
        labels,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    edge_index_like = build_review_edges(
        records,
        same_user_k=same_user_k,
        same_product_k=same_product_k,
    )
    edge_index = _edge_index_to_tensor(edge_index_like.edges)
    x = _encode_texts(texts, model_name=model_name, max_chars=max_text_chars)
    y = torch.tensor(labels, dtype=torch.long)
    unix_timestamp = torch.tensor(timestamps, dtype=torch.long)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        raw_texts=texts,
        user_ids=user_ids,
        asin=asin,
        parent_asin=parent_asin,
        unix_timestamp=unix_timestamp,
        train_mask=_indices_to_mask(train_idx, len(records)),
        val_mask=_indices_to_mask(val_idx, len(records)),
        test_mask=_indices_to_mask(test_idx, len(records)),
    )

    output_root = Path(output_root)
    sampler_dir = output_root / sampler_subdir
    output_root.mkdir(parents=True, exist_ok=True)
    sampler_dir.mkdir(parents=True, exist_ok=True)

    torch.save(data, output_root / "amazon.pt")
    torch.save({"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx, "seed": seed}, output_root / "splits.pt")

    train_batches = _materialize_neighbor_batches(data, train_idx, num_neighbors=num_neighbors, batch_size=batch_size)
    val_batches = _materialize_neighbor_batches(data, val_idx, num_neighbors=num_neighbors, batch_size=batch_size)
    test_batches = _materialize_neighbor_batches(data, test_idx, num_neighbors=num_neighbors, batch_size=batch_size)

    torch.save(train_batches, sampler_dir / "train_sampler2.pt")
    torch.save(val_batches, sampler_dir / "val_sampler2.pt")
    torch.save(test_batches, sampler_dir / "test_sampler2.pt")

    return {
        "data_path": str(output_root / "amazon.pt"),
        "splits_path": str(output_root / "splits.pt"),
        "train_sampler_path": str(sampler_dir / "train_sampler2.pt"),
        "val_sampler_path": str(sampler_dir / "val_sampler2.pt"),
        "test_sampler_path": str(sampler_dir / "test_sampler2.pt"),
        "num_nodes": len(records),
        "num_edges": len(edge_index_like.edges),
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }
