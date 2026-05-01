from src.schema.seed import SeedExample, QuestionDecompositionStep
from datasets import load_dataset


# Maps a dataset's column name to SeedExample fields.
# Key: a column name that uniquely identifies the schema.
# Value: dict mapping SeedExample field -> dataset column (None = not present).
_FIELD_MAPS: dict[str, dict] = {
    "question": {
        "instruction": "question",
        "paragraphs": "paragraphs",
        "output": "answer",
        "question_decomposition": "question_decomposition",
        "answerable": "answerable",
    },
}


class SeedLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    def load_seeds(self) -> list[SeedExample]:
        """Load seeds from a HuggingFace dataset, auto-detecting the schema."""
        dataset = load_dataset(self.dataset_name, split="train")
        cols = set(dataset.column_names)

        field_map = next(
            (m for key, m in _FIELD_MAPS.items() if key in cols), None
        )
        if field_map is None:
            raise ValueError(
                f"Cannot map columns {sorted(cols)} to SeedExample. "
                f"Add an entry to _FIELD_MAPS in seed_loader.py."
            )

        seed_list = []
        for item in dataset:
            answerable_col = field_map.get("answerable")
            if answerable_col and not item.get(answerable_col, True):
                continue

            # Paragraphs: concatenate supporting paragraphs into a string
            raw_paragraphs = item.get(field_map["paragraphs"], []) or []
            supporting = [p for p in raw_paragraphs if p.get("is_supporting", False)]
            paragraphs_text = "\n\n".join(
                f"[{p['title']}] {p['paragraph_text']}" for p in supporting
            )

            # Decomposition steps
            decomp = [
                QuestionDecompositionStep(
                    question=step["question"],
                    answer=step.get("answer", ""),
                )
                for step in item.get(field_map["question_decomposition"], [])
            ]

            seed_list.append(SeedExample(
                instruction=item[field_map["instruction"]],
                paragraphs=paragraphs_text,
                output=item[field_map["output"]],
                question_decomposition=decomp,
            ))

        return seed_list
