from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


def tokenize_log(event_log_sentences: list, variant: str, steps: int = 1) -> tuple:
    """
    Tokenize event log sentences based on the specified variant ('control-flow' or 'attributes') and
    predict the next `steps` tokens.

    Parameters:
    event_log_sentences (list): List of event log sentences.
    variant (str): Variant of the event log sentences ('control-flow' or 'attributes').
    steps (int): Number of next steps to predict.

    Returns:
    tuple: Tokenized event log sentences (xs), integer-encoded labels for the next `steps` tokens (ys),
           total number of words, maximum sequence length, and tokenizer.

    Raises:
    ValueError: If event_log_sentences is not a list or when the variant is invalid.
    """
    if not isinstance(event_log_sentences, list):
        raise ValueError("event_log_sentences must be a list")

    if variant == "control-flow":
        event_log_sentences = [
            [word for word in sentence if word.startswith("concept:name") or word in ["START==START", "END==END"]]
            for sentence in event_log_sentences
        ]
    elif variant != "attributes":
        raise ValueError("Variant not found. Please choose between 'control-flow' and 'attributes'")

    tokenizer = Tokenizer(lower=False)
    tokenizer.fit_on_texts(event_log_sentences)
    total_words = len(tokenizer.word_index) + 1

    # Print the number of unique tokens
    print(f"Number of unique tokens: {total_words - 1}")

    input_sequences = []
    ys = []

    # Generate sequences and their corresponding next steps
    for line in event_log_sentences:
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(steps, len(token_list), steps):  # Create progressively longer sequences
            # Take progressively longer context
            n_gram_sequence = token_list[:i]  # Context includes the first `i` tokens
            input_sequences.append(n_gram_sequence)

            # Collect the next `steps` tokens as targets
            next_steps = token_list[i : i + steps]
            next_steps += [0] * (steps - len(next_steps))  # Pad targets if fewer than `steps` remain
            ys.append(next_steps)

    # Pad the input sequences after generating target steps
    max_sequence_len = max(len(x) for x in input_sequences)
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre"))

    xs = input_sequences  # Properly padded input sequences
    ys = np.array(ys)

    # Print number of input sequences
    print(f"Number of input sequences: {len(xs)}")
    print(f"Sequence Length: {max_sequence_len}")

    return xs, ys, total_words, max_sequence_len, tokenizer