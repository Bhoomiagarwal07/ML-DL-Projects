# Brain Tumor (Cancer) Detection using MRI Images with CNN + Grad-CAM

## 📌 Objective
Build a Convolutional Neural Network to classify brain MRI scans as **Tumor** or **No
Tumor**, using **transfer learning** (a frozen MobileNetV2 backbone pre-trained on ImageNet)
to work effectively despite a very small dataset, and apply **Grad-CAM** to visualize which
regions of each MRI scan most influenced the model's predictions.

## 📊 Dataset
**Brain MRI Images for Brain Tumor Detection** (253 images: 155 tumor / 98 non-tumor)
Source: [Kaggle — navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

Loaded directly from a public GitHub mirror (`MohamedAliHabib/Brain-Tumor-Detection`) — no
manual upload or Kaggle API key needed.

## 🛠️ Libraries Used
- `opencv-python` (`cv2`) — image loading, resizing, color conversion, Grad-CAM overlay
- `tensorflow` / `keras` — MobileNetV2 transfer learning, model training
- `numpy` — numerical operations
- `scikit-learn` — train/test split, evaluation metrics
- `matplotlib` / `seaborn` — visualization (sample images, training curves, confusion matrix, Grad-CAM heatmaps)

## 🔍 Methodology
1. **Data Understanding** — downloaded and loaded 253 real brain MRI images, checked class
   balance (155 tumor / 98 non-tumor — moderately imbalanced), and visualized samples from
   each class.
2. **Data Preprocessing** — normalized pixel values to 0-1, split 80/20 with stratification,
   and computed class weights to counter the imbalance.
3. **Model Development** — used **transfer learning**: loaded MobileNetV2 pre-trained on
   ImageNet with its classification head removed, froze its convolutional layers, and trained
   only a new small classifier head (GlobalAveragePooling → Dense → Dropout → sigmoid output)
   on top, combined with data augmentation given the very small dataset size.
4. **Model Evaluation** — plotted training/validation curves, evaluated test accuracy,
   generated a classification report and confusion matrix.
5. **Grad-CAM Explainability** — implemented Grad-CAM to visualize which regions of each MRI
   scan most influenced the model's tumor/no-tumor prediction, overlaid as heatmaps on sample
   test images.

## 📈 Results
⚠️ This dataset is genuinely small (253 images total), so results can vary meaningfully
between runs/splits. Fill in your own run's results here, for example:

| Metric | Value |
|--------|-------|
| Test Accuracy | *(fill in from your run)* |
| Test Loss     | *(fill in from your run)* |

**Key finding:** transfer learning with a frozen pre-trained backbone substantially
outperforms training a CNN from scratch on this small dataset, since the backbone's
general-purpose visual features (edges, textures, shapes) transfer well and require far less
task-specific training data to fine-tune.

## ✅ Conclusion
This project built a transfer-learning-based CNN (using a frozen, ImageNet-pretrained
MobileNetV2 backbone) to classify brain MRI scans as Tumor or No Tumor, achieving a test
accuracy of `[fill in]` on a small 253-image dataset. Transfer learning was essential here:
with so few training images, training a CNN entirely from scratch would very likely overfit
and generalize poorly, whereas reusing general visual features already learned from millions
of ImageNet images allowed our small classifier head to learn the tumor-detection task with
far less data. We additionally applied Grad-CAM to visualize which regions of each MRI scan
most influenced the model's predictions — an essential explainability step for any medical AI
application, since a clinician needs to understand *why* a model flagged a scan as concerning,
not simply trust a black-box output. A key limitation of this project is the very small
dataset size (253 images), which limits how confidently we can generalize the model's
performance to new, unseen patient scans, and means the model should be viewed as a
proof-of-concept rather than a clinically validated diagnostic tool — a production-grade
system would need a much larger, more diverse, and clinically annotated dataset, along with
rigorous validation by medical professionals before any real-world deployment.

*(Personalize the exact test accuracy/loss figures above with your own notebook run's output.)*

## 📂 Files
- `CancerDetectionUsingMRI_GradCAM.ipynb` — full notebook with transfer learning, training, evaluation, and Grad-CAM
