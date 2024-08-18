# -*- coding: utf-8 -*-
"""3dcnn_svm_rf.ipynb

"""

from cnn_model import threedcnn
from utils import *
import pathlib
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

download_dir = pathlib.Path('paht to UCF101 dataset')

subset_paths = {
        'train': download_dir / 'train',
        'val': download_dir / 'val',
        'test': download_dir / 'test'
    }

n_frames = 10
batch_size = 8

output_signature = (tf.TensorSpec(shape = (None, None, None, 3), dtype = tf.float32),
                    tf.TensorSpec(shape = (), dtype = tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['train'], n_frames, training=True),
                                          output_signature = output_signature)


# Batch the data
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['val'], n_frames),
                                        output_signature = output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(subset_paths['test'], n_frames),
                                         output_signature = output_signature)

test_ds = test_ds.batch(batch_size)

HEIGHT = 224
WIDTH = 224

model = threedcnn(HEIGHT, WIDTH)

model.summary()

frames, label = next(iter(train_ds))
model.build(frames)
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer = keras.optimizers.Adam(learning_rate = 0.0001),
              metrics = ['accuracy'])

history = model.fit(x = train_ds,
                    epochs = 50,
                    validation_data = val_ds)

def plot_history(history):

  fig, (ax1, ax2) = plt.subplots(2)

  fig.set_size_inches(18.5, 10.5)

  # Plot loss
  ax1.set_title('Loss')
  ax1.plot(history.history['loss'], label = 'train')
  ax1.plot(history.history['val_loss'], label = 'test')
  ax1.set_ylabel('Loss')

  # Determine upper bound of y-axis
  max_loss = max(history.history['loss'] + history.history['val_loss'])

  ax1.set_ylim([0, np.ceil(max_loss)])
  ax1.set_xlabel('Epoch')
  ax1.legend(['Train', 'Validation'])

  # Plot accuracy
  ax2.set_title('Accuracy')
  ax2.plot(history.history['accuracy'],  label = 'train')
  ax2.plot(history.history['val_accuracy'], label = 'test')
  ax2.set_ylabel('Accuracy')
  ax2.set_ylim([0, 1])
  ax2.set_xlabel('Epoch')
  ax2.legend(['Train', 'Validation'])

  plt.show()

plot_history(history)

# Step 1: Create a model that outputs features instead of predictions
feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-3].output)

# Step 2: Extract features from the training, validation, and test datasets
def extract_features(dataset, feature_extractor):
    features = []
    labels = []
    for frames, label in dataset:
        extracted_features = feature_extractor(frames)
        features.append(extracted_features.numpy())
        labels.append(label.numpy())
    features = np.vstack(features)
    labels = np.concatenate(labels)
    return features, labels

# Extract features for training, validation, and test sets
train_features, train_labels = extract_features(train_ds, feature_extractor)
val_features, val_labels = extract_features(val_ds, feature_extractor)
test_features, test_labels = extract_features(test_ds, feature_extractor)

# Step 4: Train Random Forest on the extracted features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(train_features, train_labels)

# Predict and evaluate on validation set
val_predictions_rf = rf.predict(val_features)
val_accuracy_rf = accuracy_score(val_labels, val_predictions_rf)
print(f"Random Forest Validation Accuracy: {val_accuracy_rf * 100:.2f}%")

# # Evaluate on test set
test_predictions_rf = rf.predict(test_features)
test_accuracy_rf = accuracy_score(test_labels, test_predictions_rf)
print(f"Random Forest Test Accuracy: {test_accuracy_rf * 100:.2f}%")

# Compare with the original 3D CNN model
cnn_predictions = model.predict(test_ds)
cnn_accuracy = accuracy_score(test_labels, np.argmax(cnn_predictions, axis=1))
print(f"3D CNN Test Accuracy: {cnn_accuracy * 100:.2f}%")

# Function to train and evaluate SVM with different kernels
def train_and_evaluate_svm(train_features, train_labels, val_features, val_labels, test_features, test_labels, kernel_type):
    print(f"Training SVM with {kernel_type} kernel...")

    # Initialize SVM with the specified kernel
    svm = SVC(kernel=kernel_type, random_state=42)

    # Train the SVM on the extracted features
    svm.fit(train_features, train_labels)

    # Predict and evaluate on the validation set
    val_predictions = svm.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    print(f"SVM Validation Accuracy with {kernel_type} kernel: {val_accuracy * 100:.2f}%")

    # Predict and evaluate on the test set
    test_predictions = svm.predict(test_features)
    test_accuracy = accuracy_score(test_labels, test_predictions)
    print(f"SVM Test Accuracy with {kernel_type} kernel: {test_accuracy * 100:.2f}%\n")

    return val_accuracy, test_accuracy

# List of kernels to try
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

# Extract features using the feature extractor model
train_features, train_labels = extract_features(train_ds, feature_extractor)
val_features, val_labels = extract_features(val_ds, feature_extractor)
test_features, test_labels = extract_features(test_ds, feature_extractor)

# Iterate over each kernel type and evaluate the SVM
for kernel in kernels:
    train_and_evaluate_svm(train_features, train_labels, val_features, val_labels, test_features, test_labels, kernel)



