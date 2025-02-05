import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import os

# Paths
# MODEL_PATH = 'C:/Users/Sikkandhar Jabbar/Desktop/sport_sentiment_analysis/sports_sentiment_lstm_model.h5'
MODEL_PATH = 'sport_sentiment_analysis/sports_sentiment_lstm_model.h5'
DEFAULT_TEST_DATA_PATH = 'sport_sentiment_analysis/input/fifa_test_dataset.csv'

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Function to Clean Text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'http\S+', '', str(text))  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
    text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters & numbers
    text = text.lower().strip()
    return text

# Streamlit App UI
st.title("⚽ FIFA World Cup 2022 - Sentiment Analysis")
st.write("Analyze the sentiment of tweets related to the FIFA World Cup 2022.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file with tweets", type=["csv"])

if uploaded_file is not None:
    # Load Uploaded File
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
        st.success("Dataset uploaded successfully! ✅")
else:
    # Load Default Test Data
    df = pd.read_csv(DEFAULT_TEST_DATA_PATH, encoding='ISO-8859-1')
    st.info("Using default test dataset.")

# Display First Few Rows
st.subheader("📌 Preview of Dataset")
st.write(df.head())

# Show Dataset Summary
st.subheader("📊 Dataset Information")
st.write(df.info())

# Sentiment Distribution Before Prediction
st.subheader("📌 Sentiment Distribution (Before Prediction)")
st.write(df['Sentiment'].value_counts())

# Data Preprocessing
st.subheader("⏳ Preprocessing Data...")

progress_bar = st.progress(0)
df['clean_text'] = df['Tweet'].apply(clean_text)

# Tokenization & Padding
max_len = 100
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
X = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(X, maxlen=max_len)

# Encode Sentiment Labels
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])

# Simulating Processing Steps with Progress Bar
for percent_complete in range(0, 101, 10):
    time.sleep(0.1)
    progress_bar.progress(percent_complete)

st.success("Data preprocessing completed! ✅")

# Sentiment Prediction
st.subheader("🚀 Predicting Sentiment...")

# Predicting with Progress Bar Updates
predictions = []
batch_size = 32  # Set batch size for predictions
num_batches = len(X) // batch_size

for i in range(0, len(X), batch_size):
    batch = X[i:i + batch_size]
    pred = model.predict(batch)
    predictions.extend(pred)
    
    percent_complete = int((i / len(X)) * 100)
    progress_bar.progress(percent_complete)

# Convert Predictions to Sentiment Labels
df['Predicted_Sentiment'] = np.argmax(np.array(predictions), axis=1)
df['Predicted_Sentiment_Label'] = label_encoder.inverse_transform(df['Predicted_Sentiment'])

# Show Prediction Results
st.write(df[['Tweet', 'Predicted_Sentiment_Label']].head())

# Sentiment Distribution After Prediction
st.subheader("📊 Sentiment Distribution (After Prediction)")
sentiment_counts = df['Predicted_Sentiment_Label'].value_counts()
st.bar_chart(sentiment_counts)

# Pie Chart for Sentiment Distribution
fig, ax = plt.subplots()
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue'])
st.pyplot(fig)

# Download Predictions
st.subheader("📥 Download Predictions")

# Save the predictions to a CSV file
predictions_filename = "fifa_sentiment_predictions.csv"
df.to_csv(predictions_filename, index=False)

# Provide a download button for the file
st.download_button("Download Predicted Data", predictions_filename, "text/csv")

st.success("🎉 Sentiment analysis completed successfully!")

# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import re
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.text import Tokenizer
# from sklearn.preprocessing import LabelEncoder
# from wordcloud import WordCloud
# import os

# # Paths
# MODEL_PATH = 'sport_sentiment_analysis/sports_sentiment_lstm_model.h5'
# DEFAULT_TEST_DATA_PATH = 'sport_sentiment_analysis/input/fifa_test_dataset.csv'

# # Load Model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model(MODEL_PATH)

# model = load_model()

# # Function to Clean Text
# def clean_text(text):
#     if pd.isna(text):
#         return ""
#     text = re.sub(r'http\S+', '', str(text))  # Remove URLs
#     text = re.sub(r'@[A-Za-z0-9]+', '', text)  # Remove mentions
#     text = re.sub(r'#[A-Za-z0-9]+', '', text)  # Remove hashtags
#     text = re.sub(r'[^A-Za-z ]+', '', text)  # Remove special characters & numbers
#     text = text.lower().strip()
#     return text

# # Streamlit App UI
# st.title("⚽ FIFA World Cup 2022 - Sentiment Analysis")
# st.write("Analyze the sentiment of tweets related to the FIFA World Cup 2022.")

# # File Upload
# uploaded_file = st.file_uploader("Upload a CSV file with tweets", type=["csv"])

# if uploaded_file is not None:
#     # Load Uploaded File
#     with st.spinner("Loading dataset..."):
#         df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
#         st.success("Dataset uploaded successfully! ✅")
# else:
#     # Load Default Test Data
#     df = pd.read_csv(DEFAULT_TEST_DATA_PATH, encoding='ISO-8859-1')
#     st.info("Using default test dataset.")

# # Display First Few Rows
# st.subheader("📌 Preview of Dataset")
# st.write(df.head())

# # Show Dataset Summary
# st.subheader("📊 Dataset Information")
# st.write(df.info())

# # Sentiment Distribution Before Prediction
# st.subheader("📌 Sentiment Distribution (Before Prediction)")
# st.write(df['Sentiment'].value_counts())

# # Data Preprocessing
# st.subheader("⏳ Preprocessing Data...")

# progress_bar = st.progress(0)
# df['clean_text'] = df['Tweet'].apply(clean_text)

# # Tokenization & Padding
# max_len = 100
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df['clean_text'])
# X = tokenizer.texts_to_sequences(df['clean_text'])
# X = pad_sequences(X, maxlen=max_len)

# # Encode Sentiment Labels
# label_encoder = LabelEncoder()
# df['sentiment_encoded'] = label_encoder.fit_transform(df['Sentiment'])

# # Simulating Processing Steps with Progress Bar
# for percent_complete in range(0, 101, 10):
#     time.sleep(0.1)
#     progress_bar.progress(percent_complete)

# st.success("Data preprocessing completed! ✅")

# # Sentiment Prediction
# st.subheader("🚀 Predicting Sentiment...")

# # Predicting with Progress Bar Updates
# predictions = []
# batch_size = 32  # Set batch size for predictions
# num_batches = len(X) // batch_size

# for i in range(0, len(X), batch_size):
#     batch = X[i:i + batch_size]
#     pred = model.predict(batch)
#     predictions.extend(pred)
    
#     percent_complete = int((i / len(X)) * 100)
#     progress_bar.progress(percent_complete)

# # Convert Predictions to Sentiment Labels
# df['Predicted_Sentiment'] = np.argmax(np.array(predictions), axis=1)
# df['Predicted_Sentiment_Label'] = label_encoder.inverse_transform(df['Predicted_Sentiment'])

# # Show Prediction Results
# st.write(df[['Tweet', 'Predicted_Sentiment_Label']].head())

# # Sentiment Distribution After Prediction
# st.subheader("📊 Sentiment Distribution (After Prediction)")
# sentiment_counts = df['Predicted_Sentiment_Label'].value_counts()
# st.bar_chart(sentiment_counts)

# # Pie Chart for Sentiment Distribution
# fig, ax = plt.subplots()
# ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['green', 'red', 'blue'])
# st.pyplot(fig)

# # Word Cloud for Each Sentiment
# st.subheader("☁️ Word Clouds for Each Sentiment")

# sentiments = df['Predicted_Sentiment_Label'].unique()
# for sentiment in sentiments:
#     st.write(f"### {sentiment} Sentiment")
#     text = " ".join(df[df['Predicted_Sentiment_Label'] == sentiment]['clean_text'])
#     wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis('off')
#     st.pyplot(plt)

# # Sentiment Trends Over Time (if 'Date' column exists)
# if 'Date' in df.columns:
#     st.subheader("📈 Sentiment Trends Over Time")
#     df['Date'] = pd.to_datetime(df['Date'])
#     df.set_index('Date', inplace=True)
#     sentiment_over_time = df.resample('D')['Predicted_Sentiment_Label'].value_counts().unstack().fillna(0)
#     st.line_chart(sentiment_over_time)

# # Download Predictions
# st.subheader("📥 Download Predictions")

# # Save the predictions to a CSV file
# predictions_filename = "fifa_sentiment_predictions.csv"
# df.to_csv(predictions_filename, index=False)

# # Provide a download button for the file
# st.download_button("Download Predicted Data", predictions_filename, "text/csv")

# st.success("🎉 Sentiment analysis completed successfully!")
