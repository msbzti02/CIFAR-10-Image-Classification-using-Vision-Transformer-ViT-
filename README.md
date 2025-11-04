### 🧠 CIFAR-10 Image Classification using Vision Transformer (ViT)
## 🔍 Project Overview
This project fine-tunes a pre-trained Vision Transformer (ViT) model on the CIFAR-10 dataset for multi-class image classification.
It integrates transfer learning, deep learning optimization, and evaluation visualization using modern ML frameworks — Hugging Face Transformers, PyTorch, and Evaluate.

The goal is to leverage transformer-based architectures for image understanding tasks, showcasing how NLP transformer principles generalize to vision problems.

## ⚙️ Machine Learning & Deep Learning Focus
🧩 1. Transfer Learning with ViT

  - Base model: google/vit-base-patch16-224-in21k
    -  Pre-trained on ImageNet-21k with 14 million images.

  - Fine-tuning strategy:
    -  Unfreeze all layers for full fine-tuning
    - Adjust classification head to 10 CIFAR-10 classes
    - Optimize using AdamW with weight decay

## 🧠 2. Dataset Handling

  - Used Hugging Face Datasets for automatic loading and splitting of CIFAR-10.
  - Created a train/validation/test split (80/10/10).
  - Applied custom transformations using torchvision:
    - RandomResizedCrop for data augmentation
    - Normalize using model’s pretrained mean & std
    - ToTensor for PyTorch compatibility


## 🧮 Model Architecture
| Component         | Description                                      |
|-------------------|--------------------------------------------------|
| Backbone          | Vision Transformer (ViT-Base, Patch16, 224×224)  |
| Input             | 224×224 RGB images                               |
| Feature Extractor | 12 Transformer Encoder layers                    |
| Attention Heads   | 12 self-attention layers                         |
| Embedding Dim     | 768                                              |
| Output Layer      | Dense (10 units, Softmax)                        |
| Loss Function     | Cross-Entropy                                    |
| Optimizer         | AdamW with weight decay = 0.01                   |
| Learning Rate     | 2e-4                                             |
| Epochs            | 5                                                |
| Batch Size        | 32                                               |


## 📊 Evaluation & Metrics

| Metric     | Description                                         |
|------------|-----------------------------------------------------|
| Accuracy   | Proportion of correctly classified samples          |
| Precision  | True positives / (True + False positives)           |
| Recall     | True positives / (True + False negatives)           |
| F1-score   | Harmonic mean of precision and recall               |



## 📈 Visualizations
Training & Validation Curves:

1- Plotted accuracy, F1-score, and loss across epochs to monitor convergence.

<img width="716" height="463" alt="image" src="https://github.com/user-attachments/assets/adcf524a-7f4b-4a70-8400-fdb7756f0923" />
<img width="642" height="475" alt="image" src="https://github.com/user-attachments/assets/d0b85262-66d1-4556-ac63-beb291ed59a8" />
<img width="701" height="477" alt="image" src="https://github.com/user-attachments/assets/363fd9c3-7008-4bf8-9af7-de8a19c65742" />
<img width="656" height="488" alt="image" src="https://github.com/user-attachments/assets/07f45202-b52b-43b0-abd4-cc8fd251bc19" />


2- Confusion Matrix:

Computed using sklearn.metrics.confusion_matrix to visualize per-class prediction performance.
<img width="939" height="788" alt="image" src="https://github.com/user-attachments/assets/3e3ee741-a251-41f0-bbc0-8dbbf47c6708" />

3- Per-Class Accuracy Bar Chart:

Visualizes individual class accuracies (e.g., airplane, cat, ship).
<img width="988" height="584" alt="image" src="https://github.com/user-attachments/assets/aad681f3-0cec-4bbe-a44e-43a8fa5918a5" />


## 🧪 Inference & Demo

<img width="570" height="437" alt="image" src="https://github.com/user-attachments/assets/80b6730c-7d61-45e2-bb08-fd3c2b836a90" />
<img width="511" height="419" alt="image" src="https://github.com/user-attachments/assets/440b5560-dde3-4e8a-a70c-23e4f2f44d87" />
<img width="495" height="355" alt="image" src="https://github.com/user-attachments/assets/a605689c-2005-4aba-9797-6e2046d33d25" />



## 🚀 ML/DL Concepts Demonstrated
| Concept                | Description                                             |
|------------------------|---------------------------------------------------------|
| Transfer Learning      | Reusing pretrained weights for new classification tasks |
| Fine-tuning            | Optimizing model parameters on a new dataset            |
| Transformer Architecture | Multi-head self-attention for visual tokens           |
| Data Augmentation      | Improving generalization with random crops              |
| Evaluation Metrics     | Multi-metric analysis beyond accuracy                   |
| Visualization          | Loss, accuracy, F1, and confusion matrix analysis       |



🧑‍💻 Author

Mourad Sleem ibshena

moradbshina@gmail.com


