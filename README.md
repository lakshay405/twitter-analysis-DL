# twitter-analysis-DL
Text Classification with LSTM Neural Network
This project focuses on text classification using LSTM (Long Short-Term Memory) neural networks to classify textual content into specific categories. The dataset used in this project consists of labeled textual data that needs to be classified into predefined categories.

Project Overview
The project includes the following major components:

Data Loading and Preprocessing
Model Development with LSTM
Training, Evaluation, and Model Optimization
Data Loading and Preprocessing
Loading Data: The dataset is loaded from training.csv and testing.csv files using pandas. Columns are renamed for consistency, unnecessary columns are removed, and the training and testing datasets are combined into a single dataframe.
Cleaning and Preprocessing: Text data is preprocessed to remove special characters, tokenize text into words, remove stopwords, and perform lemmatization using NLTK.
Model Development with LSTM
Text Vectorization: Text data is vectorized using Keras' Tokenizer and padded to ensure uniform input size for LSTM.
LSTM Model Architecture: A Bidirectional LSTM model is built with an embedding layer, Bidirectional LSTM layer for capturing context from both directions, GlobalMaxPooling1D for dimensionality reduction, and fully connected layers with dropout for regularization.
Compilation and Training: The model is compiled with Adam optimizer, binary cross-entropy loss for binary classification, and accuracy metric. Training is performed with early stopping using a ModelCheckpoint callback to save the best model based on validation loss.
Training, Evaluation, and Model Optimization
Training: The LSTM model is trained on the preprocessed text data with specified parameters such as batch size, epochs, and learning rate.
Validation: Model performance is evaluated on a held-out validation set to monitor accuracy and loss during training.
Model Optimization: Techniques like dropout regularization and early stopping are employed to prevent overfitting and improve generalization.
Installation
To run this project, ensure Python and the necessary libraries are installed:

TensorFlow/Keras
Pandas
Matplotlib
NLTK
