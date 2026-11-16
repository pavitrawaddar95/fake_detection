
import re
import json
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

FAKE_FILE = BASE_DIR / "Fake.csv"
TRUE_FILE = BASE_DIR / "True.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_FILE = BASE_DIR / "best_fake_news_model.pkl"


# ==========================================
# 2. LOAD DATASET USING PANDAS
# ==========================================

def load_dataset():

    print("Loading datasets...")

    fake_data = pd.read_csv(FAKE_FILE)
    true_data = pd.read_csv(TRUE_FILE)

    print("Fake dataset shape:", fake_data.shape)
    print("True dataset shape:", true_data.shape)

    # Assign labels
    fake_data["label"] = 0
    true_data["label"] = 1

    # Combine datasets
    data = pd.concat(
        [fake_data, true_data],
        ignore_index=True
    )

    # Remove duplicate records
    data.drop_duplicates(inplace=True)

    # Handle missing values
    for column in ["title", "text", "subject"]:
        if column in data.columns:
            data[column] = data[column].fillna("").astype(str)
        else:
            data[column] = ""

    # Combine text columns
    data["content"] = (
        data["title"] + " " +
        data["subject"] + " " +
        data["text"]
    )

    data = data[data["content"].str.strip() != ""]

    print("\nCombined dataset shape:", data.shape)
    print("\nClass distribution:")
    print(data["label"].value_counts())

    return data


# ==========================================
# 3. TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================
# 4. TRAINING AND TESTING
# ==========================================

def train_models(data):

    X = data["content"]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))

    # Machine learning algorithms
    models = {

        "Naive Bayes": MultinomialNB(),

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42
        )
    }

    results = []
    trained_models = {}

    # Train each algorithm
    for name, classifier in models.items():

        print("\nTraining:", name)

        pipeline = Pipeline([

            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    stop_words="english",
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95
                )
            ),

            ("classifier", classifier)

        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(
            y_test, predictions
        )

        precision = precision_score(
            y_test, predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test, predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test, predictions,
            zero_division=0
        )

        results.append({

            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1

        })

        trained_models[name] = pipeline

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1 Score:", round(f1, 4))

    return results, trained_models, X_test, y_test


# ==========================================
# 5. MODEL COMPARISON
# ==========================================

def compare_models(results):

    results_df = pd.DataFrame(results)

    results_df.sort_values(
        by="F1 Score",
        ascending=False,
        inplace=True
    )

    print("\nMODEL PERFORMANCE COMPARISON")

    print(results_df.to_string(index=False))

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    return results_df


# ==========================================
# 6. SAVE BEST MODEL
# ==========================================

def save_best_model(
    results_df,
    trained_models,
    X_test,
    y_test
):

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    predictions = best_model.predict(X_test)

    joblib.dump(best_model, MODEL_FILE)

    print("\nBest Model:", best_model_name)

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Fake", "Real"],
            zero_division=0
        )
    )

    # Save metrics
    with open(
        BASE_DIR / "model_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results_df.to_dict(orient="records"),
            file,
            indent=4
        )

    return best_model, predictions


# ==========================================
# 7. CONFUSION MATRIX
# ==========================================

def create_confusion_matrix(
    y_test,
    predictions
):

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"]
    )

    plt.title("Fake News Detection Confusion Matrix")

    plt.xlabel("Predicted Label")

    plt.ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png"
    )

    plt.show()


# ==========================================
# 8. MODEL COMPARISON GRAPH
# ==========================================

def create_comparison_graph(results_df):

    graph_data = results_df.set_index("Model")[
        ["Accuracy", "Precision", "Recall", "F1 Score"]
    ]

    graph_data.plot(
        kind="bar",
        figsize=(10, 6),
        ylim=(0, 1)
    )

    plt.title("Machine Learning Model Comparison")

    plt.xlabel("Algorithms")

    plt.ylabel("Performance Score")

    plt.xticks(rotation=15)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "model_comparison.png"
    )

    plt.show()


# ==========================================
# 9. PREDICT NEWS
# ==========================================

def predict_news(model, news_text):

    prediction = model.predict([news_text])[0]

    if prediction == 0:

        print("\nPrediction: FAKE NEWS")

    else:

        print("\nPrediction: REAL NEWS")


# ==========================================
# 10. MAIN FUNCTION
# ==========================================

def main():

    print("=" * 55)

    print("FAKE NEWS DETECTION USING MACHINE LEARNING")

    print("=" * 55)

    # Load dataset
    data = load_dataset()

    # Train models
    results, trained_models, X_test, y_test = train_models(data)

    # Compare models
    results_df = compare_models(results)

    # Save best model
    best_model, predictions = save_best_model(
        results_df,
        trained_models,
        X_test,
        y_test
    )

    # Generate visualizations
    create_confusion_matrix(
        y_test,
        predictions
    )

    create_comparison_graph(results_df)

    print("\nModel training completed successfully!")

    print("Saved model:", MODEL_FILE)

    # Example prediction
    sample_news = X_test.iloc[0]

    predict_news(
        best_model,
        sample_news
    )


if __name__ == "__main__":

    main()
