import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import util

def one_hot_encode_sequences(data, num_classes):
    return [tf.keras.utils.to_categorical(list(seq), num_classes=num_classes) for _, seq in data.iterrows()]

def chunk_data(data, chunk_size):
    chunks_X = [data[i:i+chunk_size] for i in range(0, len(data)-chunk_size)]
    chunks_y = [data[i] for i in range(chunk_size, len(data))]
    return chunks_X, chunks_y

def combine_sequences(one_hot_sequences, chunk_size):
    all_X, all_y = [], []
    for sequence in one_hot_sequences:
        X, y = chunk_data(sequence, chunk_size)
        all_X.extend(X)
        all_y.extend(y)
    return np.array(all_X), np.array(all_y)

def build_lstm_model(chunk_size, num_classes):
    model = Sequential()
    model.add(LSTM(50, input_shape=(chunk_size, num_classes), activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def compile_and_train_model(model, X_train, y_train):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Load data
human_df = util.load_human_data()
num_classes = util.num_states

# One-hot encode sequences
one_hot_sequences = one_hot_encode_sequences(human_df, num_classes)

# Combine sequences
chunk_size = util.chunk_size
all_X, all_y = combine_sequences(one_hot_sequences, chunk_size)

# Split data
X_train, X_test, y_train, y_test = train_test_split(all_X, all_y, test_size=0.2, random_state=util.random_seed)

# Build and train the LSTM model
model = build_lstm_model(chunk_size, num_classes)
compile_and_train_model(model, X_train, y_train)

# Evaluate the model on the test set
evaluate_model(model, X_test, y_test)

# Save model
model.save("analysis-and-visualization\models\LSTM-RNN.h5")