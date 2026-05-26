# src/preprocessing.py
import json
import random
from datasets import Dataset


def charger_corpus(chemin: str) -> list:
    """Charge le corpus eduDS."""
    with open(chemin, "r") as f:
        data = json.load(f)

    exemples = []
    for item in data:
        for source in item["Source_Context"]:
            texte = source["page_content"]
            concepts = [c["Concept"]
                       for c in item["Core_Concepts"]]
            if len(texte) < 100:
                continue
            exemples.append({
                "sujet" : item["Knowledge_Topic"],
                "texte" : texte,
                "gold"  : concepts
            })
    return exemples


def formater(exemple: dict) -> dict:
    """Formate un exemple au format Alpaca."""
    return {
        "text": (
            "### Instruction:\n"
            "You are an expert in computer science "
            "education. Extract all technical concepts "
            "from the following text. "
            "Return ONLY this JSON format: "
            '{"concepts": ["concept1", "concept2"]}'
            "\n\n"
            f"### Input:\n{exemple['texte'][:500]}"
            "\n\n"
            "### Response:\n"
            + json.dumps(
                {"concepts": exemple["gold"]},
                ensure_ascii=False
            )
        )
    }


def preparer_splits(
    exemples: list,
    seed: int = 42
) -> dict:
    """Divise en train/val/test."""
    random.seed(seed)
    random.shuffle(exemples)
    n = len(exemples)
    return {
        "train" : exemples[:int(n * 0.8)],
        "val"   : exemples[int(n*0.8):int(n*0.9)],
        "test"  : exemples[int(n*0.9):]
    }


def creer_datasets(splits: dict) -> dict:
    """Crée les Dataset HuggingFace."""
    return {
        nom: Dataset.from_list(
            [formater(e) for e in data]
        )
        for nom, data in splits.items()
    }


if __name__ == "__main__":
    exemples = charger_corpus(
        "eduDS/docs/query/query1.json"
    )
    splits = preparer_splits(exemples)
    datasets = creer_datasets(splits)

    for nom, ds in datasets.items():
        print(f"✓ {nom:5} : {len(ds)} exemples")