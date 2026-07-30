# CIFAR-10 Image Classification using CNN

## 📌 Objective
Build and train a **Convolutional Neural Network (CNN)** from scratch to classify 32×32 color
images into 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
truck) using the classic CIFAR-10 benchmark dataset.

## 📊 Dataset
**CIFAR-10** (60,000 images: 50,000 training + 10,000 test, 10 balanced classes)
Source: [CIFAR-10 — Canadian Institute For Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)

Loaded directly via `tensorflow.keras.datasets.cifar10` — no manual download or API key needed.

## 🛠️ Libraries Used
- `tensorflow` / `keras` — building and training the CNN
- `numpy` — numerical operations
- `pandas` — class distribution table
- `matplotlib` / `seaborn` — visualization (sample images, training curves, confusion matrix, predictions)
- `scikit-learn` — classification report, confusion matrix

## 🔍 Methodology
1. **Data Understanding** — loaded CIFAR-10 via Keras, visualized sample images, confirmed
   the dataset is perfectly balanced (5,000 images/class in training, 1,000/class in test).
2. **Data Preprocessing** — normalized pixel values to the 0-1 range, flattened label arrays
   for use with sparse categorical crossentropy, and built a data augmentation pipeline
   (random horizontal flip, rotation, zoom) applied only during training.
3. **Model Development** — built a CNN with 3 convolutional blocks (each with Conv2D →
   BatchNormalization → Conv2D → MaxPooling → Dropout), followed by dense layers, trained
   with the Adam optimizer and EarlyStopping (up to 25 epochs).
4. **Model Evaluation** — plotted training/validation accuracy and loss curves, evaluated test
   accuracy, generated a full classification report and confusion matrix, and visualized
   sample predictions.

## 📈 Results
⚠️ Training a CNN takes real compute time and benefits significantly from a GPU. Enable
**Runtime → Change runtime type → T4 GPU** in Colab before running. Fill in your own run's
results here, for example:

| Metric | Value |
|--------|-------|
| Test Accuracy | *(fill in from your run — typically ~75-82% for this architecture)* |
| Test Loss     | *(fill in from your run)* |

**Expected pattern:** vehicle classes (automobile, ship, truck, airplane) are usually
classified more reliably than visually similar animal classes (cat, dog, bird, deer), which
share overlapping textures and shapes at CIFAR-10's low 32×32 resolution.

## ✅ Conclusion
This project built and trained a Convolutional Neural Network from scratch to classify
CIFAR-10 images into 10 categories, using a 3-block convolutional architecture with batch
normalization, dropout, and data augmentation to improve generalization. The model achieved a
test accuracy of `[fill in]`, with vehicle classes (automobile, ship, truck, airplane)
generally classified more reliably than visually similar animal classes (cat, dog, bird,
deer), which tend to share overlapping textures and shapes at CIFAR-10's low 32×32 resolution.
Data augmentation and dropout played an important role in keeping the model from overfitting
to the training set, as reflected in how closely the training and validation accuracy/loss
curves track each other. Key limitations of this approach include the relatively small image
size (which limits how much fine-grained detail the model can learn from) and the fact that a
compact CNN trained from scratch, without transfer learning from a larger pre-trained model
(such as ResNet or EfficientNet), will generally underperform state-of-the-art approaches that
leverage features learned from much larger image datasets.

*(Personalize the exact test accuracy/loss figures above with your own notebook run's output.)*

## 📂 Files
- `Cifar10ImageClassificationUsingCNN.ipynb` — full notebook with CNN architecture, training, and evaluation
