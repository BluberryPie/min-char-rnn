# min-char-rnn

A character-level recurrent neural network implemented from scratch in NumPy, inspired by Andrej Karpathy's [min-char-rnn](https://gist.github.com/karpathy/d4dee566867f8291f086). No deep learning frameworks, just NumPy.

## Requirements

- Python 3.9+
- `numpy`
- `tqdm`

Install with:

```bash
pip install numpy tqdm
```

## Project structure

```
config.py   # Hyperparameters and file paths
model.py    # RNN class: forward pass, backprop, Adagrad update, save/load, sampling
train.py    # Training loop
sample.py   # Generate text from a trained model
data/       # Training corpora (e.g. shakespeare.txt)
resources/  # Saved model weights and char<->index mappings
```

## Usage

### Train a model

```bash
python train.py
```

This reads the text file at `config.input_file` (defaults to `data/shakespeare.txt`), trains the RNN for `config.num_epochs` epochs, and saves:

- Model weights to `resources/model.npz`
- Character-to-index mapping to `resources/char_to_idx.pkl`

Use `--debug` for verbose logging:

```bash
python train.py --debug
```

### Sample text from a trained model

```bash
python sample.py
```

Loads the saved model and mapping, then generates and prints `config.sample_length` characters starting from the seed character `"A"`.

## Configuration

All hyperparameters live in `config.py`:

| Field | Description | Default |
|---|---|---|
| `input_file` | Path to training text | `data/shakespeare.txt` |
| `hidden_size` | Size of the RNN hidden state | `100` |
| `learning_rate` | Adagrad learning rate | `0.01` |
| `num_epochs` | Number of training epochs | `10` |
| `seq_length` | Sequence length per training step | `25` |
| `model_savepath` | Where to save trained weights | `resources/model.npz` |
| `char_to_idx_path` | Where to save the char/index mapping | `resources/char_to_idx.pkl` |
| `sample_length` | Number of characters to sample | `500` |
| `batch_size` | Number of parallel sequences per training step | `32` |

## Notes

This project is primarily a learning exercise for understanding how RNNs, backpropagation through time. Everything is implemented without autograd or deep learning libraries such as PyTorch.
