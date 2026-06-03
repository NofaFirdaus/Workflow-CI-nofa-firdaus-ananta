import os
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_keras
import mlflow
import mlflow.tensorflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)


parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--batch_size",    type=int,   default=8)
parser.add_argument("--epochs",        type=int,   default=3)
parser.add_argument("--max_len",       type=int,   default=128)
args = parser.parse_args()

MODEL_NAME    = "indobenchmark/indobert-base-p1"
NUM_LABELS    = 3
RANDOM_STATE  = 42
LEARNING_RATE = args.learning_rate
BATCH_SIZE    = args.batch_size
EPOCHS        = args.epochs
MAX_LEN       = args.max_len

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "hok_preprocessing.csv"
)

dagshub.auth.add_app_token(token=os.environ["DAGSHUB_TOKEN"])

dagshub.init(repo_owner='NofaFirdaus', repo_name='sistem-machine-learning', mlflow=True)


print("[INFO] Memuat dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text_slangwords", "sentiment"])

le = LabelEncoder()
df["label"] = le.fit_transform(df["sentiment"])
label_names  = le.classes_.tolist()
print(f"[INFO] Label mapping: { {v: k for k, v in enumerate(label_names)} }")
print(f"[INFO] Distribusi kelas:\n{df['sentiment'].value_counts()}")

X = df["text_slangwords"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=RANDOM_STATE, stratify=y_train
)

print(f"[INFO] Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")


print("[INFO] Memuat tokenizer IndoBERT...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(
        list(texts),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="tf"
    )

print("[INFO] Tokenisasi data...")
train_enc = tokenize(X_train)
val_enc   = tokenize(X_val)
test_enc  = tokenize(X_test)

def to_tf_dataset(encodings, labels, batch_size, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "input_ids":      encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings["token_type_ids"],
        },
        labels
    ))
    if shuffle:
        dataset = dataset.shuffle(1000)
    return dataset.batch(batch_size)

train_ds = to_tf_dataset(train_enc, y_train, BATCH_SIZE, shuffle=True)
val_ds   = to_tf_dataset(val_enc,   y_val,   BATCH_SIZE)
test_ds  = to_tf_dataset(test_enc,  y_test,  BATCH_SIZE)


print("[INFO] Memuat model IndoBERT...")
model = TFBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS
)

optimizer = tf_keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss_fn   = tf_keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=["accuracy"])


with mlflow.start_run(run_name="IndoBERT_CI"):

    mlflow.log_param("model_name",    MODEL_NAME)
    mlflow.log_param("max_len",       MAX_LEN)
    mlflow.log_param("batch_size",    BATCH_SIZE)
    mlflow.log_param("epochs",        EPOCHS)
    mlflow.log_param("learning_rate", LEARNING_RATE)
    mlflow.log_param("num_labels",    NUM_LABELS)
    mlflow.log_param("train_size",    len(X_train))
    mlflow.log_param("val_size",      len(X_val))
    mlflow.log_param("test_size",     len(X_test))
    mlflow.log_param("optimizer",     "Adam")
    mlflow.log_param("loss",          "SparseCategoricalCrossentropy")

    print("[INFO] Mulai training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)


    for epoch in range(EPOCHS):
        mlflow.log_metric("train_loss",     history.history["loss"][epoch],         step=epoch)
        mlflow.log_metric("train_accuracy", history.history["accuracy"][epoch],     step=epoch)
        mlflow.log_metric("val_loss",       history.history["val_loss"][epoch],     step=epoch)
        mlflow.log_metric("val_accuracy",   history.history["val_accuracy"][epoch], step=epoch)

    print("[INFO] Evaluasi test set...")
    y_pred_logits = model.predict(test_ds).logits
    y_pred = np.argmax(y_pred_logits, axis=1)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    mlflow.log_metric("test_accuracy",           acc)
    mlflow.log_metric("test_precision_weighted", prec)
    mlflow.log_metric("test_recall_weighted",    rec)
    mlflow.log_metric("test_f1_weighted",        f1)

    print(f"\nTest Accuracy  : {acc:.4f}")
    print(f"Test Precision : {prec:.4f}")
    print(f"Test Recall    : {rec:.4f}")
    print(f"Test F1        : {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=label_names, yticklabels=label_names, ax=ax
    )
    ax.set_title("Confusion Matrix - IndoBERT CI")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)
    os.remove(cm_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"],     label="Train Loss")
    axes[0].plot(history.history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
    axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle("Training History - IndoBERT CI")
    plt.tight_layout()
    hist_path = "training_history.png"
    plt.savefig(hist_path)
    plt.close()
    mlflow.log_artifact(hist_path)
    os.remove(hist_path)

    report = classification_report(y_test, y_pred, target_names=label_names, zero_division=0)
    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)
    os.remove(report_path)

    model_dir = "indobert_model"
    model.save_pretrained(model_dir)
    mlflow.log_artifacts(model_dir, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    with open("run_id.txt", "w") as f:
        f.write(run_id)
    print(f"\n[INFO] run_id: {run_id}")
    print("[INFO] Training selesai. Artefak tersimpan di DagsHub MLflow.")