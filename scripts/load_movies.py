import json
from typing import Any, Dict, Set

from datasets import load_dataset

NUM_ROWS_TO_SAVE = 1000
DATA_FILE = "data/movies.json"


def valid_examples(example: Dict[str, Any]) -> bool:
    return (
        example["original_language"] == "en"
        and example["genres"]
        and example["overview"]
        and example["title"]
        and example["release_date"]
    )


unique_ids: Set[Any] = set()


def is_unique(example: Dict[str, Any]) -> bool:
    if example["id"] in unique_ids:
        return False
    else:
        unique_ids.add(example["id"])
        return True


def convert_to_document(example: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(example["id"]),
        "content": example["overview"],
        "meta": {
            "title": example["title"],
            "runtime": example["runtime"],
            "vote_average": example["vote_average"],
            "release_date": example["release_date"],
            "genres": example["genres"].split("-"),
        },
    }


movies_dataset = (
    load_dataset("wykonos/movies", split="train")
    .filter(valid_examples)
    .filter(is_unique)
    .shuffle(seed=42)
    .select(range(NUM_ROWS_TO_SAVE))
)

movie_documents = [convert_to_document(movie) for movie in movies_dataset]

with open(DATA_FILE, "w") as outfile:
    json.dump(movie_documents, outfile)
