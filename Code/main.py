import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_score, recall_score

# Function to load data from a CSV file
def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    labels = data.iloc[:, 0].values  # First column is the label
    images = data.iloc[:, 1:].values  # Remaining columns are pixel values
    return images, labels

# Step 1: Load the dataset
train_images, train_labels = load_data_from_csv('mnist_train.csv')
test_images, test_labels = load_data_from_csv('mnist_test.csv')

# Normalize the pixel values to [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape images to include the channel dimension (28x28x1)
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Convert labels to one-hot encoding
train_labels = np.eye(10)[train_labels]
test_labels = np.eye(10)[test_labels]

print(f"Train images shape: {train_images.shape}")
print(f"Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Step 2: Define the Neural Network Architecture
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Explicit Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Print the model summary
model.summary()

# Step 3: Train the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

# Step 4: Model Evaluation
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred = model.predict(test_images)
y_pred_classes = y_pred.argmax(axis=1)
y_test_classes = test_labels.argmax(axis=1)

# Classification Metrics
accuracy = test_accuracy
precision = precision_score(y_test_classes, y_pred_classes, average='macro')
sensitivity = recall_score(y_test_classes, y_pred_classes, average='macro')

print(f"Classification Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")

print("\nClassification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Plot ROC Curves
plt.figure(figsize=(12, 8))
for i in range(10):  # 10 classes (digits 0-9)
    fpr, tpr, _ = roc_curve(test_labels[:, i], y_pred[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Digit {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.title("ROC Curves for Each Class")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Visualize Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.show()

# Experimentation 1: Add Batch Normalization
model_bn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Explicit Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model_bn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_bn = model_bn.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

test_loss_bn, test_accuracy_bn = model_bn.evaluate(test_images, test_labels)
y_pred_bn = model_bn.predict(test_images)

print(f"Batch Normalization Test Accuracy: {test_accuracy_bn:.4f}")

# Experimentation 2: Adjust Dropout Rate
model_dropout = models.Sequential([
    layers.Input(shape=(28, 28, 1)),  # Explicit Input layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),  # Adjusted Dropout rate
    layers.Dense(10, activation='softmax')
])

model_dropout.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_dropout = model_dropout.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

test_loss_dropout, test_accuracy_dropout = model_dropout.evaluate(test_images, test_labels)
y_pred_dropout = model_dropout.predict(test_images)

print(f"Dropout Adjustment Test Accuracy: {test_accuracy_dropout:.4f}")

# Visualization and Comparison
# Plot comparison of accuracy over experiments
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label="Initial (Baseline)", linestyle="--")
plt.plot(history_bn.history['accuracy'], label="With Batch Normalization")
plt.plot(history_dropout.history['accuracy'], label="Dropout Adjusted (0.3)")
plt.title("Training Accuracy Comparison")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Plot ROC curves for comparison
plt.figure(figsize=(12, 8))
fpr, tpr, _ = roc_curve(test_labels[:, 0], y_pred[:, 0])  # Initial
plt.plot(fpr, tpr, label="Initial ROC (Digit 0)")
fpr, tpr, _ = roc_curve(test_labels[:, 0], y_pred_bn[:, 0])  # BatchNorm
plt.plot(fpr, tpr, label="BatchNorm ROC (Digit 0)")
fpr, tpr, _ = roc_curve(test_labels[:, 0], y_pred_dropout[:, 0])  # Dropout
plt.plot(fpr, tpr, label="Dropout Adjusted ROC (Digit 0)")
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve Comparison for Digit 0")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
