import curses
import tensorflow as tf
import numpy as np
import util

num_classes = util.num_states
chunk_size = util.chunk_size

model = tf.keras.models.load_model("analysis-and-visualization\models\LSTM-RNN.h5")

def clean_text(seq):
    return " ".join(map(str, seq))

def get_prediction(input_seq):
    if len(input_seq) >= 5:
        model_seq = tf.keras.utils.to_categorical(input_seq[-5:], num_classes=num_classes).reshape((1, chunk_size, num_classes))
        next_pred = np.argmax(model.predict(model_seq))
        next_pred_prob = np.max(model.predict(model_seq))
        return next_pred, next_pred_prob
    else:
        return None, None

def display_prediction(stdscr, next_pred, next_pred_prob):
    stdscr.addstr(3, 0, "Prediction:")
    if next_pred is not None:
        stdscr.addstr(4, 0, f"Next char: {next_pred} Confidence: {round(next_pred_prob * 100, 1)}%")
    else:
        stdscr.addstr(4, 0, "Next char: ? Confidence: ?")

def main(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.clear()

    # Initialize variable
    input_seq = []

    while True:
        stdscr.clear()

        stdscr.addstr(0, 0, "Type sequence: ")
        stdscr.addstr(1, 0, clean_text(input_seq))

        next_pred, next_pred_prob = get_prediction(input_seq)
        display_prediction(stdscr, next_pred, next_pred_prob)

        # Get user input
        key = stdscr.getch()

        # Update input text and letter count
        if 48 <= key <= 58:  # ASCII printable characters
            input_seq.append(chr(key))

curses.wrapper(main)