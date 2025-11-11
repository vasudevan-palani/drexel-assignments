import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau


# -------------------------------------------------
# 0. Config
# -------------------------------------------------
MODEL_PATH = "cifar10_model.h5"
EPOCHS = 10
BATCH_SIZE = 64

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# -------------------------------------------------
# 1. Load CIFAR-10
# -------------------------------------------------
print("Loading CIFAR-10...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# normalize
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")

# -------------------------------------------------
# 2. Build model (simple CNN)
# -------------------------------------------------
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# -------------------------------------------------
# 3. Train (or load if exists)
# -------------------------------------------------
if os.path.exists(MODEL_PATH) and False:
    print(f"Found existing model at {MODEL_PATH}, loading it...")
    model = load_model(MODEL_PATH)
    history = None  # we don't have training history when loading
else:
    print("Building and training a new model...")
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    model = build_model()
    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[lr_scheduler],
        verbose=1
    )
    # save after training
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# -------------------------------------------------
# 4. Evaluate on test set
# -------------------------------------------------
print("Evaluating on test data...")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

# -------------------------------------------------
# 5. Predictions & metrics
# -------------------------------------------------
print("Getting predictions for metrics...")
y_proba = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_proba, axis=1)

print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# -------------------------------------------------
# 6. Confusion matrix (with matplotlib heatmap)
# -------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # normalize?
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_confusion_matrix(cm, class_names, title="CIFAR-10 Confusion Matrix")

# -------------------------------------------------
# 7. Training curves (only if we trained now)
# -------------------------------------------------
if history is not None:
    # accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# -------------------------------------------------
# 8. Show some sample predictions
# -------------------------------------------------
def show_sample_predictions(num_samples=5):
    plt.figure(figsize=(12, 3))
    for i in range(num_samples):
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[y_pred[idx]]

        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(f"T:{true_label}\nP:{pred_label}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

show_sample_predictions(5)

# -------------------------------------------------
# 9. Demo: predict a single test image with top-3
# -------------------------------------------------
idx = 0
sample = x_test[idx:idx+1]
probs = model.predict(sample, verbose=0)[0]
top3 = np.argsort(probs)[-3:][::-1]

print("\nSingle image demo:")
print("True label:", class_names[y_test[idx]])
for i in top3:
    print(f"  {class_names[i]} -> {probs[i]:.4f}")

print("\nDone.")
