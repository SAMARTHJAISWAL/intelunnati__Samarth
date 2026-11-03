# Conquering Fashion-MNIST with CNNs

Clothing image classification with a small convolutional neural network (TensorFlow/Keras) on the
[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. Built as a project for the
Intel Unnati programme (MIT Manipal).

## Dataset

**Fashion-MNIST** (Zalando Research): 70,000 grayscale 28×28 images of clothing in 10 classes
(T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). There are 60,000
training images and 10,000 test images.

- The code downloads the dataset automatically with `tf.keras.datasets.fashion_mnist.load_data()`, so you don't need to download anything by hand.
- `data/fashion-mnist_test.csv` (22 MB, 10,000 rows of `label, pixel1 … pixel784`) is the CSV copy of the test split from the [Kaggle Fashion-MNIST dataset](https://www.kaggle.com/datasets/zalando-research/fashionmnist). The code does not read it. It is kept for reference.
- `data/train-data` contains a Google Drive link to the matching training CSV, which is too large for the repo.

## Approach

1. Reshape the images to `28×28×1` and scale pixel values to `[0, 1]`.
2. CNN: `Conv2D(64, 3×3, ReLU) → MaxPool(2) → Conv2D(64, 3×3, ReLU) → MaxPool(2) → Flatten → Dense(128, ReLU) → Dense(10, softmax)`. The network has 243,786 parameters.
3. Train with Adam and sparse categorical cross-entropy for 5 epochs (batch size 32). The test split is passed as `validation_data`, so it is used for monitoring only. No early stopping or tuning is done.
4. Evaluate with test accuracy, a confusion matrix, a per-class classification report, one-vs-rest ROC curves with AUC, and macro precision/recall/F1.

## Results

Results from the notebook `model/MNIST.ipynb` (re-executed for this README with TensorFlow 2.21 and `set_random_seed(42)`, 5 epochs, CPU):

| Metric (10,000 test images) | Value |
|---|---|
| Test accuracy | **90.49%** |
| Macro precision / recall / F1 | 0.905 / 0.905 / 0.904 |
| Best class (F1) | Trouser, Sandal, Bag: 0.98 |
| Hardest class (F1) | Shirt: 0.72 |

The original Colab run (in this repo's git history) reached **90.74%** test accuracy.

> The Word documents in `doc/` state a final accuracy of 93%. No run in this repo shows that figure. The notebooks show 90.5–90.7% on the test set; 93.2% is the *training* accuracy after epoch 5.

### Notebook changes

- Removed the `!pip install visualkeras` cell and the no-op `tf.onednn=1` line.
- Used `tf.keras.Input` instead of `input_shape=` (deprecated in Keras 3).
- Added a small shim so `visualkeras` can draw the model under Keras 3.
- Added class names to the plots and report.
- Replaced the precision/recall cell. It fed integer class labels to the *binary* `tf.keras.metrics.Precision/Recall` and reported misleading ~0.98 values. It now reports sklearn macro averages.
- Removed the "IoU" cells. IoU is a segmentation metric. Here it was computed between a one-hot label and a probability vector and always came out at 0.10, which says nothing about the model.

## How to run

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cd "samarth_MIT_Manipal_ Conquering Fashion MNIST with CNNs using Computer Vision"
python code/MNIST.py          # trains, prints metrics, saves plots + model to code/outputs/
# or
jupyter notebook model/MNIST.ipynb
```

Training takes about 1.5–2.5 minutes on a laptop CPU. The script writes `fashion_mnist_cnn.keras`, `model_diagram.png`, `confusion_matrix.png`, `roc_curves.png` and `accuracy.png` to `code/outputs/`.

## Repository structure

```
samarth_MIT_Manipal_ Conquering Fashion MNIST with CNNs using Computer Vision/
├── code/MNIST.py               # script version: train, evaluate, save plots and model
├── model/MNIST.ipynb           # notebook with outputs (training log, plots, report)
├── data/
│   ├── fashion-mnist_test.csv  # test split as CSV (reference only)
│   └── train-data              # Google Drive link to the training CSV
├── doc/
│   ├── REPORT.docx             # project report (conclusion, tests, accuracy)
│   └── approach.docx           # approach and reasoning
└── demo-vid/DEMO VIDEO         # Google Drive link to the demo video
requirements.txt
```

The demo video is hosted on Google Drive:
<https://drive.google.com/file/d/14TK81LNRQeXTgmDmJ5oFDYaemkhbened/view?usp=sharing>

## Tech stack

Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, Matplotlib, seaborn, visualkeras, Jupyter.
