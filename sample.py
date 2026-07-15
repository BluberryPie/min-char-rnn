import pickle
import sys
import time

from config import Config
from model import RNN


if __name__ == "__main__":
    config = Config()

    # Load the model from the specified path
    model = RNN.from_file(config.model_savepath)

    # Load the char_to_idx mapping from the specified path -> create idx_to_char mapping
    with open(config.char_to_idx_path, "rb") as f:
        char_to_idx = pickle.load(f)
    idx_to_char = {v: k for k, v in char_to_idx.items()}

    # Sample text from the trained model
    starting_char = "A"
    sampled_text = model.sample(seed_idx=char_to_idx[starting_char],
                                sample_length=config.sample_length)
    sampled_text = "".join(idx_to_char[idx] for idx in sampled_text)
    
    # Print the sampled text character by character with a delay(just for fun)
    for char in sampled_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)