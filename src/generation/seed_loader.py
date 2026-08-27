"""Loads real multi-hop examples from a HuggingFace dataset to seed generation, detecting which
column layout the dataset uses and normalizing every row into a SeedExample.
"""

import json

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.schema.seed import QuestionDecompositionStep, SeedExample

# Datasets whose HuggingFace loading script is incompatible with datasets>=3.0.
# Maps dataset name to (repo file path, repo type); the raw JSON is downloaded directly.
_RAW_HF_FILES: dict[str, tuple[str, str]] = {
    "alabnii/morehopqa": ("data/with_human_verification.json", "dataset"),
}

# Maps a dataset's column layout onto SeedExample fields, keyed by a column that
# uniquely identifies that layout. More specific layouts are listed first.
# paragraphs_type "list" means dicts with title/paragraph_text/is_supporting;
# "context_list" means [[title, [sentences...]], ...].
_FIELD_MAPS: dict[str, dict] = {
    # alabnii/morehopqa, identified by "previous_question"
    "previous_question": {
        "instruction": "question",
        "paragraphs": "context",
        "paragraphs_type": "context_list",
        "output": "answer",
        "question_decomposition": "question_decomposition",
        "answerable": None,
    },
    # MuSiQue and other multi-hop sets with structured paragraphs
    "question": {
        "instruction": "question",
        "paragraphs": "paragraphs",
        "paragraphs_type": "list",
        "output": "answer",
        "question_decomposition": "question_decomposition",
        "answerable": "answerable",
    },
}


class SeedLoader:
    """Loads and normalizes seed examples from one HuggingFace dataset."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_seeds(self) -> list[SeedExample]:
        """Load every usable seed example from the dataset.

        Returns:
            The normalized seeds, skipping rows the dataset marks unanswerable.

        Raises:
            ValueError: If the dataset is empty or its columns are unrecognized.
        """
        raw_items = self._load_raw_items()
        if not raw_items:
            raise ValueError(f"Dataset {self.dataset_name} returned no rows")

        field_map = self._find_field_map(set(raw_items[0].keys()))
        seeds = [self._to_seed(item, field_map) for item in raw_items]
        return [seed for seed in seeds if seed is not None]

    def _load_raw_items(self) -> list[dict]:
        """Fetch the dataset rows as plain dicts.

        Returns:
            The raw rows, read from a downloaded JSON file when the dataset's
            loading script is unusable, otherwise through the datasets library.
        """
        if self.dataset_name in _RAW_HF_FILES:
            file_path, repo_type = _RAW_HF_FILES[self.dataset_name]
            local_path = hf_hub_download(self.dataset_name, file_path, repo_type=repo_type)
            with open(local_path) as f:
                return json.load(f)
        return list(load_dataset(self.dataset_name, split="train"))

    @staticmethod
    def _find_field_map(columns: set[str]) -> dict:
        """Detect which column layout the dataset uses.

        Params:
            columns: Column names of the first row.

        Returns:
            The matching field map.

        Raises:
            ValueError: If no known layout matches.
        """
        for key, field_map in _FIELD_MAPS.items():
            if key in columns:
                return field_map
        raise ValueError(
            f"Cannot map columns {sorted(columns)} to SeedExample. "
            f"Add an entry to _FIELD_MAPS in seed_loader.py."
        )

    @classmethod
    def _to_seed(cls, item: dict, field_map: dict) -> SeedExample | None:
        """Normalize one dataset row.

        Params:
            item: The raw row.
            field_map: The layout detected for this dataset.

        Returns:
            The seed example, or None if the dataset marks the row unanswerable.
        """
        answerable_column = field_map.get("answerable")
        if answerable_column and not item.get(answerable_column, True):
            return None

        return SeedExample(
            instruction=item[field_map["instruction"]],
            paragraphs=cls._paragraphs_text(item, field_map),
            output=item[field_map["output"]],
            question_decomposition=[
                QuestionDecompositionStep(question=step["question"], answer=step.get("answer", ""))
                for step in item.get(field_map["question_decomposition"], [])
            ],
        )

    @staticmethod
    def _paragraphs_text(item: dict, field_map: dict) -> str:
        """Flatten a row's supporting paragraphs into text.

        Params:
            item: The raw row.
            field_map: The layout detected for this dataset.

        Returns:
            The supporting paragraphs, one titled block each.
        """
        raw_paragraphs = item.get(field_map["paragraphs"]) or []
        if field_map.get("paragraphs_type") == "context_list":
            # Entries are [title, [sentences...]]; index rather than unpack so a
            # dataset carrying extra fields per entry still loads.
            return "\n\n".join(
                f"[{entry[0]}] {''.join(entry[1])}" for entry in raw_paragraphs
            )
        return "\n\n".join(
            f"[{p['title']}] {p['paragraph_text']}"
            for p in raw_paragraphs
            if p.get("is_supporting", False)
        )
