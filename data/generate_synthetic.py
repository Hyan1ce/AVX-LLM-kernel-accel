import torch
import numpy as np

def generate_synthetic_data(vocab_size=10000, seq_len=128, batch_size=32, num_samples=1000):
    """Generate synthetic data for language modeling."""
    data = []
    for _ in range(num_samples):
        # Generate random token sequences
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
        data.append(tokens)
    return data

if __name__ == "__main__":
    print("Generating synthetic data...")
    data = generate_synthetic_data()
    print(f"Generated {len(data)} samples")
    print(f"Each sample shape: {data[0].shape}")

