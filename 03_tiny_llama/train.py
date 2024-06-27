# trains a TinyLlama model to generate endless WallStreetBets comments
# download dataset from https://www.kaggle.com/datasets/nikolaimelnikov/rwallstreetbets-comments-1120-till-1130

from model import Config, TinyLlama
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from tqdm import tqdm, trange
import random

BATCH_SIZE = 512
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1_000_000_000
VAL_EXAMPLES = 5000  # how many comments to leave for validation
VAL_FREQ = 500  # how many batches to fit before validating

def main():
    device = get_device()
    config = Config()
    model = TinyLlama(config).to(device)
    train_text = get_train_text()
    tokenizer = train_tokenizer(train_text, config)
    train_ids = tokenize_train_text(train_text, tokenizer, config)
    random.shuffle(train_ids)
    val_ids = train_ids[:VAL_EXAMPLES]
    train_ids = train_ids[VAL_EXAMPLES:]
    train(model, train_ids, val_ids, device)

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"  # apple silicon
    else:
        device = "cpu"
    print("Torch Device:", device)
    torch.set_default_device(device)
    return device

def get_train_text() -> List[str]:
    print("Preprocessing comments...")
    train_text = []
    for filename in ["archive/comments_2020-11-1_2020-11-30_new.csv", "archive/comments_2020-12-1_2020-12-31_new.csv", "archive/comments_2021-1-1_2021-11-29_new.csv"]:
        df = pd.read_csv(filename)
        for i in range(len(df)):
            text = " " + str(df.iloc[i, 3]).strip()
            text = text.encode("ascii", "ignore").decode("ascii")  # ignore all non-ascii characters
            train_text.append("[BOS]" + text + "[EOS]")
    return train_text

def train_tokenizer(text: List[str], config: Config) -> Tokenizer:
    # train a byte-pair encoding tokenizer
    print("Training tokenizer...")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=config.vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"])
    def iterator():
        for i in range(0, len(text), 1000):
            yield text[i:i + 1000]
    tokenizer.train_from_iterator(iterator(), trainer=trainer)
    tokenizer.save("tokenizer.json")
    return tokenizer

def tokenize_train_text(train_text: List[str], tokenizer: Tokenizer, config: Config) -> List[List[int]]:
    print("Tokenizing all text...")
    train_ids = []
    for i in tqdm(range(len(train_text))):
        ids = tokenizer.encode(train_text[i]).ids[:config.context_length]
        if len(ids) < config.context_length:
            ids.extend([0] * (config.context_length - len(ids)))
        assert len(ids) == config.context_length
        train_ids.append(ids)
    return train_ids

def train(model: TinyLlama, train_ids: List[List[int]], val_ids: List[List[int]], device: str):
    model.mode = "train"
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = 10000
    for epoch_num in range(NUM_EPOCHS):
        for step in (tqdm_t := trange(0, len(train_ids), BATCH_SIZE)):
            token_ids = train_ids[step:step + BATCH_SIZE]
            token_ids = torch.Tensor(token_ids).int().to(device)
            logits = model.forward(token_ids, 0)
            preds = F.softmax(logits, dim=-1)  # (batch_size, seq_len, vocab_size)
            labels = torch.cat((token_ids[:, 1:], torch.zeros(token_ids.shape[0], 1).int()), dim=1)  # (batch_size, seq_len) labels is just token_ids shifted in time by one unit
            labels[:, :1] = 0  # [BOS] -> predict next token is too hard task, don't count in loss
            b, i = torch.where(labels)
            loss = -preds[b, i, labels[b, i]].log2().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tqdm_t.set_description(f"Epoch: {epoch_num+1} | Loss: {loss.item()}")
            if (step // BATCH_SIZE) % VAL_FREQ == 0:  # whether to validate
                with torch.no_grad():
                    sum_val_loss = 0
                    num_val_loss = 0
                    for val_step in tqdm(range(0, len(val_ids), BATCH_SIZE)):
                        token_ids = val_ids[val_step:val_step + BATCH_SIZE]
                        token_ids = torch.Tensor(token_ids).int().to(device)
                        logits = model.forward(token_ids, 0)
                        preds = F.softmax(logits, dim=-1)
                        labels = torch.cat((token_ids[:, 1:], torch.zeros(token_ids.shape[0], 1).int()), dim=1)
                        labels[:, :1] = 0  # [BOS] -> predict next token is too hard task, don't count in loss
                        b, i = torch.where(labels)
                        loss = -preds[b, i, labels[b, i]].log2().mean()
                        sum_val_loss += loss.item()
                        num_val_loss += 1
                    val_loss = sum_val_loss / num_val_loss
                    print(f"Val Loss: {val_loss}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(model, "WallStreetBets_model.pt")
                        print("Model saved!")
        torch.save(model, "WallStreetBets_model_overfit.pt")
        print("Overfit model saved!")

if __name__ == "__main__":
    main()
