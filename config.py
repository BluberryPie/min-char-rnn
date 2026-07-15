from dataclasses import dataclass


@dataclass
class Config:
    input_file: str = "data/shakespeare.txt"
    hidden_size: int = 100
    learning_rate: float = 0.01
    num_epochs: int = 10
    seq_length: int = 25
    model_savepath: str = "resources/model.npz"
    char_to_idx_path: str = "resources/char_to_idx.pkl"
    sample_length:int = 500