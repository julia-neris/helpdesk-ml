import json
from pathlib import Path
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent

def main():
    rows = json.loads((ROOT / "data/tickets.json").read_text(encoding="utf-8"))
    groups = {split: {row["group"] for row in rows if row["split"] == split}
              for split in ("train", "validation", "test")}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"]
                or groups["test"] & groups["validation"])
    def subset(split):
        chosen = [row for row in rows if row["split"] == split]
        return [row["text"] for row in chosen], [row["label"] for row in chosen]
    x_train, y_train = subset("train")
    x_val, y_val = subset("validation")
    x_test, y_test = subset("test")
    models = {
        "majority": make_pipeline(TfidfVectorizer(), DummyClassifier(strategy="most_frequent")),
        "word_tfidf": make_pipeline(TfidfVectorizer(ngram_range=(1, 2)),
                                   LogisticRegression(max_iter=1000, random_state=42)),
        "char_tfidf": make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)),
                                   LogisticRegression(max_iter=1000, random_state=42)),
    }
    validation = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        validation[name] = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
    best_name = max(validation, key=validation.get)
    best = models[best_name]
    labels = sorted(set(y_train))
    prediction = best.predict(x_test)
    result = {
        "source": "80 mensagens sintéticas, 40 cenários",
        "split_counts": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "selection": "macro-F1 na validação; modelo mantido com treino original",
        "validation_macro_f1": validation, "selected": best_name,
        "test_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
        "labels": labels, "confusion_matrix": confusion_matrix(y_test, prediction, labels=labels).tolist(),
        "predictions": [dict(text=text, expected=expected, predicted=predicted)
                        for text, expected, predicted in zip(x_test, y_test, prediction)],
    }
    output = ROOT / "model"
    output.mkdir(exist_ok=True)
    joblib.dump(best, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best_name, "test_macro_f1": result["test_report"]["macro avg"]["f1-score"]}))

if __name__ == "__main__":
    main()
