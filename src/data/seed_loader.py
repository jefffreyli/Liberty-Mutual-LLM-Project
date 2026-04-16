from src.schema.seed import SeedExample
from datasets import load_dataset

class SeedLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
    def load_seeds(self) -> list[SeedExample]:
        """
        Load the seed from existing dataset from Hugging Face.
        """
        dataset = load_dataset(self.dataset_name, split="train")
        seed_list = []
        for item in dataset:
            seed_list.append(SeedExample(
                instruction=item["instruction"],
                input=item["input"],
                output=item["output"],
            ))
        return seed_list