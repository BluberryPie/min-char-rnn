import numpy as np


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Vector dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # Model parameters
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))
        # Memory variables for Adagrad
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mWhy = np.zeros_like(self.Why)
        self.mbh = np.zeros_like(self.bh)
        self.mby = np.zeros_like(self.by)
    
    def forward(self, inputs: list[list[int]], targets: list[list[int]], h_prevs: np.ndarray):
        # Get the batch/sequence size from the inputs
        batch_size = len(inputs)
        sequence_length = len(inputs[0])  # Assumes all sequences in the batch have the same length
        
        # One-hot encode the inputs (Each xs[t] is a 2D array of shape (input_size, batch_size))
        xs:dict[int, np.ndarray] = dict()
        for t in range(sequence_length):
            xs[t] = np.zeros((self.input_size, batch_size))
            for b in range(batch_size):
                xs[t][inputs[b][t], b] = 1  # One-hot encoding for each sequence in the batch

        # Forward pass through the RNN
        loss: float = 0.0
        hs: dict[int, np.ndarray] = dict()
        ps: dict[int, np.ndarray] = dict()
        hs[-1] = h_prevs
        for t, x in xs.items():
            hs[t] = np.tanh(self.Wxh @ x + self.Whh @ hs[t - 1] + self.bh)
            y = self.Why @ hs[t] + self.by
            # Numerical stability for softmax
            numerator = np.exp(y - np.max(y, axis=0))
            denominator = np.sum(np.exp(y - np.max(y, axis=0)), axis=0)
            ps[t] = numerator / denominator
            target_indices = np.array([targets[b][t] for b in range(batch_size)])
            loss += -np.log(ps[t][target_indices, np.arange(batch_size)]).sum()  # Cross-entropy loss for the batch
        
        return xs, hs, ps, loss

    def backward(self, xs, hs, ps, targets):
        batch_size = len(targets)
        
        dy: dict[int, np.ndarray] = dict()
        dby = np.zeros_like(self.by)
        dWhy = np.zeros_like(self.Why)
        dh_next = np.zeros((self.hidden_size, batch_size))  # Grad from next time step
        dbh = np.zeros_like(self.bh)
        dWhh = np.zeros_like(self.Whh)
        dWxh = np.zeros_like(self.Wxh)
        
        for t in reversed(xs):
            # Compute gradient for logits(y)
            dy[t] = ps[t]
            for b in range(batch_size):
                dy[t][targets[b][t], b] -= 1
            # Compute gradient for by and Why
            dby += dy[t].sum(axis=1, keepdims=True)
            dWhy += dy[t] @ hs[t].T
            # Compute gradient for bh, Whh, Wxh
            dh = self.Why.T @ dy[t] + dh_next
            dh_raw = (1 - hs[t] ** 2) * dh  # Grad through tanh
            dbh += dh_raw.sum(axis=1, keepdims=True)
            dWhh += dh_raw @ hs[t - 1].T
            dWxh += dh_raw @ xs[t].T
            # Update dh_next for next iteration
            dh_next = self.Whh.T @ dh_raw

        # Gradient clipping to prevent exploding gradients
        for gradient in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(gradient, -5, 5, out=gradient)
        
        return dWxh, dWhh, dWhy, dbh, dby

    def update(self, dWxh, dWhh, dWhy, dbh, dby, lr):
        # Update model parameters using Adagrad
        for param, dparam, mem in zip(
            [self.Wxh, self.Whh, self.Why, self.bh, self.by],
            [dWxh, dWhh, dWhy, dbh, dby],
            [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]
        ):
            mem += dparam * dparam
            param += -lr * dparam / (np.sqrt(mem) + 1e-8)
    
    def save_model(self, file_path):
        np.savez(file_path, Wxh=self.Wxh, Whh=self.Whh, Why=self.Why, bh=self.bh, by=self.by)
    
    @classmethod
    def from_file(cls, file_path):
        # Load model parameters from a file
        data = np.load(file_path)
        input_size = data['Wxh'].shape[1]
        hidden_size = data['Wxh'].shape[0]
        output_size = data['Why'].shape[0]
        # Create a new RNN instance and set its parameters
        model = cls(input_size, hidden_size, output_size)
        model.Wxh = data['Wxh']
        model.Whh = data['Whh']
        model.Why = data['Why']
        model.bh = data['bh']
        model.by = data['by']
        
        return model
    
    def sample(self, seed_idx: int, sample_length: int):
        # Sample a sequence of indices from the model
        x = np.zeros((self.input_size, 1))
        x[seed_idx] = 1
        h = np.zeros((self.hidden_size, 1))
        indices = []
        
        for t in range(sample_length):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            idx = np.random.choice(range(self.output_size), p=p.ravel())
            indices.append(idx)
            # Prepare the next input
            x = np.zeros((self.input_size, 1))
            x[idx] = 1
        
        return indices