import argparse
import os

from http2vec.evaluation import *
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import os



def get_args():
    parser = argparse.ArgumentParser(
        description="Language model for HTTP"
    )

    parser.add_argument(
        "output_path", help="Output for train model"
    )

    parser.add_argument(
        "--size", help="Model name",
    )
    parser.add_argument(
        "--lm", help="Language model"
    )
    parser.add_argument(
        "--clf-name", help="Language model"
    )
    parser.add_argument("--dataset")
    parser.add_argument("--against")

    return parser.parse_args()


def main():

    args = get_args()
    os.makedirs(f"{args.output_path}/saved", exist_ok=True)

    # get classify function
    vectorizer = get_vectorizer(
        language_model=args.lm,
        dataset=args.dataset,
        size=args.size,
    )

    # get data to be classified (any split really)
    data = get_data(
        dataset=args.against
    )
    train_ids, test_ids = get_split(
        dataset=args.against,
        language_model=args.lm,
        size=args.size
    )

    tokenizer = RobertaTokenizer.from_pretrained(
        f"data/tokenizers/{args.dataset}",
    )
    clf = RandomForestClassifier(random_state=0, n_jobs=-1)
    clf.fit(
        [vectorizer.vectorize(doc.splitlines(), tokenizer) for doc in data.loc[train_ids]["text"]],
        data.loc[train_ids]["label"]
    )

    classify = get_classifier_fn(
        vectorizer=vectorizer,
        tokenizer_path=f"data/tokenizers/{args.dataset}",
        classifier=clf,
        mode="predict"
    )
    data = data.loc[test_ids]

    true_y = list(data["label"])
    pred_y = classify(list(data["text"]))

    f1 = f1_score(true_y, pred_y, pos_label="anomaly")
    print("F1", f1)
    predictions_save = {
        _id: l for _id, l in zip(data["id"], pred_y)
    }
    with open(f"{args.output_path}/saved/predictions-cross.json", "wt") as f:
        json.dump(predictions_save, f)
    with open(f"{args.output_path}/metrics.json", "wt") as f:
        json.dump({"f1": f1}, f)


if __name__ == "__main__":
    main()
