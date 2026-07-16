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

    # Prompt the user for a starting character and validate it
    sys.stdout.write("Enter a starting character: ")
    sys.stdout.flush()
    starting_char = sys.stdin.read(1)
    if starting_char not in char_to_idx:
        print(f"Character '{starting_char}' not found in the model's vocabulary.")
        sys.exit(1)

    # Sample text from the trained model
    sampled_text = model.sample(seed_idx=char_to_idx[starting_char],
                                sample_length=config.sample_length)
    sampled_text = starting_char + "".join(idx_to_char[idx] for idx in sampled_text)
    
    # Print the sampled text character by character with a delay(just for fun)
    for char in sampled_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.01)
    sys.stdout.write("\n")