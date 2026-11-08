"""Fashion-MNIST classification with a small CNN (TensorFlow / Keras).

Script version of ../model/MNIST.ipynb. It trains the model for 5 epochs,
prints test accuracy and a classification report, and writes the plots
(model diagram, confusion matrix, ROC curves, accuracy curve) and the trained
model into an ``outputs/`` folder next to this script.

Usage:
    python MNIST.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files; no display needed

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from tensorflow.keras import datasets

OUT_DIR = Path(__file__).resolve().parent / "outputs"
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1) / 255.0
    x_test = x_test.reshape(10000, 28, 28, 1) / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def save_model_diagram(model, path):
    try:
        import visualkeras
    except ImportError:
        print("visualkeras not installed; skipping model diagram")
        return
    # visualkeras 0.2 reads layer.output_shape, which Keras 3 removed.
    for layer in model.layers:
        if not hasattr(layer, "output_shape"):
            layer.output_shape = tuple(layer.output.shape)
    visualkeras.layered_view(model).save(path)


def main():
    tf.keras.utils.set_random_seed(42)
    OUT_DIR.mkdir(exist_ok=True)

    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()
    model.summary()

    history = model.fit(x_train, y_train, epochs=5,
                        validation_data=(x_test, y_test), verbose=2)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("Test Accuracy:", test_acc)
    model.save(OUT_DIR / "fashion_mnist_cnn.keras")
    save_model_diagram(model, OUT_DIR / "model_diagram.png")

    probs = model.predict(x_test, verbose=0)
    preds = np.argmax(probs, axis=1)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png")
    plt.close()

    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    print(f"Macro precision {precision:.4f}  recall {recall:.4f}  F1 {f1:.4f}")

    # One-vs-rest ROC curves
    plt.figure()
    for i, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve((y_test == i).astype(int), probs[:, i])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", fontsize="small")
    plt.savefig(OUT_DIR / "roc_curves.png")
    plt.close()

    # Training curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim([0.5, 1])
    plt.legend(loc="lower right")
    plt.savefig(OUT_DIR / "accuracy.png")
    plt.close()

    print(f"Saved model and plots to {OUT_DIR}")


if __name__ == "__main__":
    main()
