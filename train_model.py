# train_model.py
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
import pickle

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Load IMDb dataset
(train_data, test_data), info = tfds.load(
    'imdb_reviews',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True
)

train_sentences, train_labels = [], []
for sentence, label in tfds.as_numpy(train_data):
    train_sentences.append(sentence.decode('utf-8'))
    train_labels.append(label)

test_sentences, test_labels = [], []
for sentence, label in tfds.as_numpy(test_data):
    test_sentences.append(sentence.decode('utf-8'))
    test_labels.append(label)

# Tokenization and padding
vocab_size = 20000
max_len = 200
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)

x_train = pad_sequences(tokenizer.texts_to_sequences(train_sentences), maxlen=max_len)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_sentences), maxlen=max_len)
y_train = np.array(train_labels)
y_test = np.array(test_labels)

# Build LSTM model
model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(128),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train model
history = model.fit(
    x_train, y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.2,
    verbose=2
)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\n✅ Test Accuracy: {test_acc:.4f}")

# Save model and tokenizer
model.save("model.h5")
print("✅ Model saved as model.h5")

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved as tokenizer.pkl")

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_plot_custom.png")
print("📊 Training plot saved as 'training_plot_custom.png'.")
