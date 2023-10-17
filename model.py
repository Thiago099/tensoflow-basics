import tensorflow as tf
import numpy as np
def train(x,y):


    # Create a vocabulary and tokenization
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        return lowercase

    tokenizer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization
    )

    tokenizer.adapt(x + y)
    vocab_size = len(tokenizer.get_vocabulary())

    # Tokenize the x and y
    question_tokens = tokenizer(x)
    answer_tokens = tokenizer(y)

    # Pad or truncate sequences to ensure equal length
    max_seq_length = max(len(question_tokens[0]), len(answer_tokens[0]))
    question_tokens = tf.keras.preprocessing.sequence.pad_sequences(question_tokens, maxlen=max_seq_length, padding='post', truncating='post')
    answer_tokens = tf.keras.preprocessing.sequence.pad_sequences(answer_tokens, maxlen=max_seq_length, padding='post', truncating='post')

    # Create a simple RNN model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.SimpleRNN(64, return_sequences=True),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(vocab_size, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # Train the model
    model.fit(question_tokens, answer_tokens, epochs=100, verbose=False)

    def predict(test_question):
        test_question_tokens = tokenizer(test_question)
        test_question_tokens = tf.keras.preprocessing.sequence.pad_sequences(test_question_tokens, maxlen=max_seq_length, padding='post', truncating='post')
        predicted_answers_tokens = model.predict(test_question_tokens, verbose=False)
        predicted_answers_index = np.argmax(predicted_answers_tokens, axis=-1)  # Use axis=-1 to get the indices along the last dimension
        # Convert the predicted answer indices back to words
        return [" ".join(tokenizer.get_vocabulary()[index] for index in predicted_answer_index if index > 0) for predicted_answer_index in predicted_answers_index]
    
    return predict