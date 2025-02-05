import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset with a fix for encoding issues
file_path = 'sport_sentiment_analysis/input/fifa_world_cup_2022_tweets.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fixed encoding issue

# Display dataset information
print(df.head())
print(df.info())

# Check sentiment distribution
print(df['Sentiment'].value_counts())

# Text Preprocessing
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters and numbers
    text = text.lower().strip()
    return text

# Apply text cleaning
df['clean_text'] = df['Tweet'].apply(clean_text)

# Encode Sentiment Labels
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])

# Tokenization with padding
max_len = 100
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=max_len)

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, df['sentiment_encoded'], test_size=0.2, random_state=42)

# Build LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_len),
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')  # Assuming 3 sentiment classes
])

# Compile the Model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=5,
    batch_size=32
)

# Save the Model
model.save('C:/Users/Sikkandhar Jabbar/Desktop/sport_sentiment_analysis/sports_sentiment_lstm_model.h5')

# Plot Training History
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()
