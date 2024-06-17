import pandas as pd

# Load the datasets
train_data = pd.read_csv('training.csv')
test_data = pd.read_csv('testing.csv')

# Rename columns for consistency
train_data.columns = ['ID', 'Company', 'Label', 'Content']
test_data.columns = ['ID', 'Company', 'Label', 'Content']

# Remove unnecessary columns
train_data.drop(columns=["ID", "Company"], inplace=True)
test_data.drop(columns=["ID", "Company"], inplace=True)

# Combine training and testing datasets
full_data = pd.concat([train_data, test_data], ignore_index=True)

# Check for missing values
print(full_data.isnull().sum())

# Drop rows with missing values
full_data.dropna(inplace=True)

# Check for any remaining missing values
print(full_data.isnull().sum())

# Check for duplicate rows
print(full_data.duplicated().sum())

# Remove duplicate rows
full_data.drop_duplicates(inplace=True)

# Confirm removal of duplicates
print(full_data.duplicated().sum())

# Import necessary libraries for text preprocessing
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download necessary NLTK data files
nltk.download('wordnet', "/kaggle/working/nltk_data/")
nltk.download('omw-1.4', "/kaggle/working/nltk_data/")
!unzip /kaggle/working/nltk_data/corpora/wordnet.zip -d /kaggle/working/nltk_data/corpora
!unzip /kaggle/working/nltk_data/corpora/omw-1.4.zip -d /kaggle/working/nltk_data/corpora
nltk.data.path.append("/kaggle/working/nltk_data/")

# Define a function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text, flags=re.I)  # Remove extra whitespace
    text = re.sub(r'\W', ' ', str(text))  # Remove special characters
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text)  # Remove single characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    
    tokens = word_tokenize(text)  # Tokenize the text
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    filtered_tokens = [token for token in filtered_tokens if len(token) > 3]  # Remove short words
    
    unique_indices = np.unique(filtered_tokens, return_index=True)[1]
    cleaned_tokens = np.array(filtered_tokens)[np.sort(unique_indices)].tolist()
    
    return cleaned_tokens

# Separate features and labels
X = full_data.drop('Label', axis=1)
y = full_data['Label']

# Preprocess the text data
text_data = list(X['Content'])
processed_texts = [preprocess_text(text) for text in text_data]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(processed_texts, y, test_size=0.2, random_state=42)

# Prepare the data for the model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_vocab_size = 20000
tokenizer = Tokenizer(num_words=max_vocab_size)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

print(y.value_counts())

# Build the LSTM model
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, Input, GlobalMaxPooling1D, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

embedding_dim = 100
learning_rate = 0.0001

# Define the model architecture
model = Sequential()
model.add(Input(shape=(max_sequence_length,)))
model.add(Embedding(max_vocab_size, embedding_dim))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(GlobalMaxPooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model with a checkpoint callback
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_data=(X_test_pad, y_test), callbacks=[checkpoint])
