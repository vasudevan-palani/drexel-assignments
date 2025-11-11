# 🧠 CIFAR-10 Supervised Learning Demo

A simple **Convolutional Neural Network (CNN)** built with **TensorFlow / Keras** to classify images from the **CIFAR-10 dataset** into 10 categories.  
This project demonstrates a complete **supervised learning workflow** — loading data, training a model, evaluating accuracy, and visualizing results.

---

## 📘 Overview

**Dataset:** CIFAR-10 (60,000 color images, 32×32 pixels)  
**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck  
**Task:** Multi-class image classification  
**Model type:** CNN (Conv2D → MaxPooling → Dense → Softmax)

---

## ⚙️ Requirements

### 1. Install Python 3.13 or newer
Check your version:
```bash
python --version
```

### 2. Install pip (if not available)
```bash
python -m ensurepip --upgrade
```

### 3. Install required packages

install everything from a **requirements.txt** file:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Save the main file as **main.py**
2. Run it from the terminal:
   ```bash
   python main.py
   ```
3. The program will automatically:
   - Download the CIFAR-10 dataset (if not cached)
   - Build and train the CNN
   - Evaluate on the test set
   - Print metrics (accuracy, loss, precision/recall etc.)
   - Plot training curves and show sample predictions
   - Save the trained model as **cifar10_model.h5**

---

## 🧩 Key Steps in the Program

| Step | Description |
|------|--------------|
| **Load Data** | Loads CIFAR-10 and normalizes pixel values to `[0,1]`. |
| **Build Model** | CNN with Conv2D → MaxPool → Dense → Softmax layers. |
| **Train Model** | 10 epochs, batch = 64, 10 % validation split. |
| **Evaluate** | Prints accuracy / loss / classification report / confusion matrix. |
| **Visualize** | Shows accuracy & loss curves and 5 sample predictions. |
| **Save Model** | Saves trained model for reuse (`cifar10_model.h5`). |
| **Reload Model** | `load_model("cifar10_model.h5")` for later inference. |

---

## 📈 Expected Output

During training:
```
Epoch 1/10  accuracy: 0.47  val_accuracy: 0.57
...
Epoch 10/10 accuracy: 0.95  val_accuracy: 0.75
```

Final results:
```
Test accuracy: ~0.75
Classification report printed
Confusion matrix plotted
```

Typical plots produced:
- Training vs Validation Accuracy  
- Training vs Validation Loss  
- Sample predictions with True / Predicted labels

---

## 💾 Reuse the Model

To predict later:
```python
from tensorflow.keras.models import load_model
import tensorflow as tf, numpy as np

model = load_model("cifar10_model.h5")
(_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_test = x_test.astype("float32") / 255.0

pred = model.predict(np.expand_dims(x_test[0], axis=0))
print("Predicted class index:", np.argmax(pred))
```

---

## 🧮 Metrics & Observations

| Metric | Meaning | Example |
|---------|----------|----------|
| **Training Accuracy** | Fit on training data | ≈ 95 % |
| **Validation Accuracy** | Generalization on unseen data | ≈ 75 % |
| **Observation** | Model learns quickly but starts overfitting after ~epoch 7–8. |

*Conclusion:* The CNN effectively classifies unseen images with ~75 % accuracy, demonstrating a successful supervised learning system.  
Further improvements could include data augmentation, dropout, and transfer learning.

---

## 👨‍💻 Author

**Vasu Palani**  
Graduate Student — Applied AI and Machine Learning  
