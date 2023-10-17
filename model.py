import tensorflow as tf
import numpy as np

def train(trainingData):
    
    combined_text = " ".join([x+"\0" for x in trainingData])

    # Create a set of unique characters from the combined text
    chars = sorted(set(combined_text))
    char_to_index = {char: i for i, char in enumerate(chars)}
    index_to_char = {i: char for i, char in enumerate(chars)}
    data = [char_to_index[char] for char in combined_text]


    # Create training sequences with a stop character
    seq_length = 3
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])

    # Prepare the data
    x = np.array(sequences)
    y = np.array(data[seq_length:])

    # Define the model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(len(chars), 8),
        tf.keras.layers.SimpleRNN(32),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(len(chars), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Train the model
    model.fit(x, y, epochs=1000, verbose=False)


    def predict(seed):
        # Generate text using the trained model with a stop character
        seed_text = seed
        generated_text = seed_text
        stop_character = '\0'  # Define the stop character
        num_chars_to_generate = 20

        while generated_text[-1] != stop_character and len(generated_text) < num_chars_to_generate:  # You can specify a maximum length as well
            seed_sequence = [char_to_index[char] for char in generated_text[-seq_length:]]
            next_char_index = np.argmax(model.predict(np.array(seed_sequence).reshape(1, -1), verbose=False))
            next_char = index_to_char[next_char_index]
            generated_text += next_char

        return generated_text
    
    return predict