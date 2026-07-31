# Face Recognition using CNN — Labeled Faces in the Wild (LFW)

## 📌 Objective
Build and train a **Convolutional Neural Network (CNN)** to recognize and classify faces of
well-known public figures using the **Labeled Faces in the Wild (LFW)** dataset — a classic
"in the wild" benchmark where photos are taken under real, uncontrolled conditions (varying
lighting, pose, background, and expression).

## 📊 Dataset
**Labeled Faces in the Wild (LFW)**, filtered to individuals with ≥70 photos
Source: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)

Loaded directly via `sklearn.datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)` —
no manual download or API key needed.

## 🛠️ Libraries Used
- `scikit-learn` — LFW dataset loader, train/test split, class weight computation, evaluation metrics
- `tensorflow` / `keras` — building and training the CNN
- `numpy` / `pandas` — numerical operations and class distribution table
- `matplotlib` / `seaborn` — visualization (sample faces, training curves, confusion matrix, predictions)

## 🔍 Methodology
1. **Data Understanding** — loaded LFW via scikit-learn (filtered to people with ≥70 photos),
   visualized sample "in the wild" faces, and checked the class distribution (imbalanced,
   even after filtering).
2. **Data Preprocessing** — normalized pixel values to 0-1, added a channel dimension for
   Conv2D compatibility, and split 80/20 with stratification.
3. **Model Development** — built a CNN with 3 convolutional blocks (Conv2D → BatchNormalization
   → MaxPooling → Dropout), data augmentation (flip, rotation, zoom, brightness), computed
   balanced class weights to counter imbalance, and trained with Adam + EarlyStopping.
4. **Model Evaluation** — plotted training/validation curves, evaluated test accuracy,
   generated a per-person classification report and confusion matrix, and visualized sample
   predictions.

## 📈 Results
⚠️ Training benefits from a GPU (enable **Runtime → Change runtime type → T4 GPU** in Colab).
Fill in your own run's results here, for example:

| Metric | Value |
|--------|-------|
| Test Accuracy | *(fill in from your run — typically ~75-85% for this architecture)* |
| Test Loss     | *(fill in from your run)* |

**Expected pattern:** the most frequently photographed individual is usually recognized most
reliably; misclassifications tend to occur between people with visually similar general
appearance.

## ✅ Conclusion
This project built and trained a Convolutional Neural Network to recognize faces of public
figures using the Labeled Faces in the Wild (LFW) dataset, restricted to individuals with at
least 70 photos to ensure a viable multi-class classification problem. The model achieved a
test accuracy of `[fill in]`, using data augmentation and class weighting to address the
dataset's relatively small size and class imbalance. As expected for an "in the wild" dataset,
the model performed best on the most frequently photographed individuals and showed more
confusion between people with visually similar features, reflecting the genuine difficulty of
face recognition under real-world variation in pose, lighting, and expression — as opposed to
a clean, studio-controlled dataset. One key limitation of this from-scratch CNN approach is
that, with relatively few training images per person, the model is more prone to overfitting
than it would be with a much larger training set, and would likely benefit significantly from
transfer learning using a face-specific pre-trained network (such as FaceNet or a model
pre-trained on VGGFace2) rather than learning facial features entirely from scratch on this
comparatively small dataset.

*(Personalize the exact test accuracy/loss figures above with your own notebook run's output.)*

## 📂 Files
- `FaceRecognitionUsingCNN_LFW.ipynb` — full notebook with CNN architecture, training, and evaluation
