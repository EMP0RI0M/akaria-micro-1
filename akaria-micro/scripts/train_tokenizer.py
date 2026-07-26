import os
import sys
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--save_path", type=str, default="tinystories_tokenizer.json")
    args = parser.parse_args()

    print("Loading TinyStories dataset (train split)...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    def batch_iterator():
        for i in range(0, len(dataset), 1000):
            yield dataset[i : i + 1000]["text"]

    print(f"Training BPE Tokenizer with vocab size {args.vocab_size}...")
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
        vocab_size=args.vocab_size
    )

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    tokenizer.save(args.save_path)
    print(f"Tokenizer saved to {args.save_path}")

if __name__ == "__main__":
    main()
