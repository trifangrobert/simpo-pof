from datasets import load_dataset
from datasets import Dataset
from typing import Tuple


def load_preference_data(n_train: int = 10000, n_test: int = 1000) -> Tuple[Dataset, Dataset]:
    train = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    test = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")

    train = train.filter(lambda x: x["score_chosen"] > x["score_rejected"])
    test = test.filter(lambda x: x["score_chosen"] > x["score_rejected"])

    train = train.select(range(min(n_train, len(train))))
    test = test.select(range(min(n_test, len(test))))

    train = train.select_columns(["prompt", "chosen", "rejected", "score_chosen", "score_rejected"])
    test = test.select_columns(["prompt", "chosen", "rejected", "score_chosen", "score_rejected"])

    return train, test

if __name__ == "__main__":
    train, test = load_preference_data()
    # print(train[0])
    print(train[0].keys())
    print(len(train[0]["chosen"]))
    print(train[0]["chosen"][0].keys())
    print(f"Chosen conversation:")
    print(train[0]["chosen"][0])
    print(train[0]["chosen"][1])
    print(f"-" * 50)
    print(len(train[0]["rejected"]))
    print(train[0]["rejected"][0].keys())
    print(f"Rejected conversation:")
    print(train[0]["rejected"][0])
    print(train[0]["rejected"][1])

    # print(test[0])