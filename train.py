import argparse
import logging
import pickle

import numpy as np
from tqdm import tqdm

from config import Config
from model import RNN


if __name__ == "__main__":
    config = Config()

    # Set up argument parser for command-line options
    parser = argparse.ArgumentParser(description="Train an RNN model.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Modify the input file path as needed
    with open(config.input_file, "r") as f:
        raw_text = f.read()
    
    # Create a set of unique characters and calculate the vocabulary size
    chars = list(set(raw_text))
    vocab_size = len(chars)
    logging.debug(f"Vocabulary size: {vocab_size}")

    # Create character to index mappings
    char_to_idx = {char: i for i, char in enumerate(chars)}
    logging.debug(f"Character to index mapping: {char_to_idx}")

    model = RNN(
        input_size=vocab_size,
        hidden_size=config.hidden_size,
        output_size=vocab_size)
    
    # Divide the raw text into partitions for batch processing
    partition_size = (len(raw_text) - 1) // config.batch_size  # -1 to ensure no OOB
    partition_starting_offsets = [i * partition_size for i in range(config.batch_size)]
    logging.debug(f"{partition_starting_offsets=}")

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        h_prevs = np.zeros((model.hidden_size, config.batch_size))

        pbar = tqdm(range(partition_size // config.seq_length),
                    desc=f"Epoch {epoch + 1}/{config.num_epochs}", unit="chunk")
        for chunk in pbar:
            # Inputs and Targets are now lists of lists, where each inner list corresponds to a batch
            inputs: list[list[int]] = []
            targets: list[list[int]] = []
            for offset in partition_starting_offsets:
                start_idx = offset + chunk * config.seq_length
                end_idx = start_idx + config.seq_length
                inputs.append([char_to_idx[char] for char in raw_text[start_idx:end_idx]])
                targets.append([char_to_idx[char] for char in raw_text[start_idx + 1:end_idx + 1]])

            # Forward pass through the RNN
            xs, hs, ps, loss = model.forward(inputs, targets, h_prevs)
            h_prevs = hs[config.seq_length - 1]
            epoch_loss += loss
            pbar.set_postfix(loss=f"{loss:.4f}")

            # Backward pass to compute gradients
            dWxh, dWhh, dWhy, dbh, dby = model.backward(xs, hs, ps, targets)

            # Update model parameters using the computed gradients
            model.update(dWxh, dWhh, dWhy, dbh, dby, config.learning_rate)
        
        logging.info(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {epoch_loss:.4f}")
    
    # Save the trained model parameters to a file
    logging.debug(f"Saving model parameters to {config.model_savepath}")
    model.save_model(config.model_savepath)

    # Save the character mappings to a file for later use
    logging.debug(f"Saving character to index mapping to {config.char_to_idx_path}")
    with open(config.char_to_idx_path, "wb") as f:
        pickle.dump(char_to_idx, f)