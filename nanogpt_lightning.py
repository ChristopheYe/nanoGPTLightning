# Rewrite the ng-video-lecture code from Andrej Karpathy with Lightning
# framework for multi-gpu training
# https://github.com/karpathy/ng-video-lecture

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb

wandb.init(settings=wandb.Settings(console="off"))


class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        # -1 because we need to create a full block_size+1 sequence to get block_size inputs and targets
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        # Get the input sequence and the subsequent character as the target
        chunk = self.data[idx : idx + self.block_size + 1]
        input_seq = chunk[:-1]
        target_seq = chunk[1:]
        return input_seq, target_seq


class nanoGPT_data(L.LightningDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 64, block_size: int = 128):
        super().__init__()
        self.save_hyperparameters()

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        # only create the data once
        pass

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        with open(self.hparams.data_dir, "r") as f:
            self.text = f.read()
        chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.text = torch.tensor([self.stoi[c] for c in self.text], dtype=torch.long)
        n1 = int(0.8 * len(self.text))
        n2 = int(0.9 * len(self.text))

        if stage == "fit" or stage is None:
            self.train_dataset = CharDataset(self.text[:n1], self.hparams.block_size)
            self.val_dataset = CharDataset(self.text[n1:n2], self.hparams.block_size)

        if stage == "test" or stage is None:
            self.test_dataset = CharDataset(self.text[n2:], self.block_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size)


class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.2):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        q = self.query(x)  # (B,T,hs)
        k = self.key(x)  # (B,T,hs)
        v = self.value(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        weights = (
            q @ k.transpose(-2, -1) / (C**0.5)
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weights = F.softmax(weights, dim=-1)  # (B, T, T)
        # perform the weighted aggregation of the values
        out = weights @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )  ##DD1
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.Tanh(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.2):
        super().__init__()
        head_size = n_embd // n_head
        self.ma = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd, dropout)
        self.ln2 = nn.LayerNorm(n_embd, dropout)

    def forward(self, x):
        x = x + self.ma(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class GPTLanguageModel(L.LightningModule):
    def __init__(
        self, n_embd, n_head, n_layer, vocab_size, block_size, dropout, learning_rate
    ):
        super().__init__()
        self.save_hyperparameters()
        self.token_embedding_table = nn.Embedding(
            self.hparams.vocab_size, self.hparams.n_embd
        )
        self.position_embedding_table = nn.Embedding(
            self.hparams.block_size, self.hparams.n_embd
        )
        self.blocks = nn.Sequential(
            *[
                Block(
                    self.hparams.n_embd,
                    self.hparams.n_head,
                    self.hparams.block_size,
                    self.hparams.dropout,
                )
                for _ in range(n_layer)
            ]
        )  # DD2
        self.ln_f = nn.LayerNorm(self.hparams.n_embd)
        self.lm_head = nn.Linear(self.hparams.n_embd, self.hparams.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device)  # DD3
        )  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,V)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        self.log("train_loss", loss)
        print("train_loss : ", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        self.log("val_loss", loss)
        print("val_loss : ", loss)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

    def generate(self, idx, max_new_tokens):
        self.to(idx.device)  # DD4
        # idx is (B, T) array of indices in the current context
        results = []  # To store the output indices for decoding later
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.trainer.datamodule.hparams.block_size :]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
            # Append the newly generated index, squeeze to flatten
            results.append(idx_next.squeeze(1))

        # Decode the results
        decoded_text = "".join(
            [
                self.trainer.datamodule.itos[int(i)]
                for i in torch.cat(results).cpu().numpy()
            ]
        )
        return decoded_text


def main():
    # hyperparameters
    batch_size = 64  # (B) how many independent sequences will we process in parallel?
    block_size = 256  # (T) what is the maximum context length for predictions?
    vocab_size = 67  # (V) how big is our vocabulary
    max_iters = 5000  # how many training iterations
    learning_rate = 3e-4
    n_embd = 384  # (C) how many dimensions per token
    n_head = 6  # how many attention heads?
    n_layer = 6  # how many layers?
    dropout = 0.2  # what dropout rate to use?
    data_dir = "input.txt"
    output_path = "/home2/cye73/StatisticalML/nanoGPT"
    experiment_name = "nanoGPT"
    # ------------

    data_module = nanoGPT_data(
        data_dir=data_dir, batch_size=batch_size, block_size=block_size
    )

    # model = GPTLanguageModel(
    #     n_embd=n_embd,
    #     n_head=n_head,
    #     n_layer=n_layer,
    #     vocab_size=vocab_size,
    #     block_size=block_size,
    #     dropout=dropout,
    #     learning_rate=learning_rate,
    # )

    model = GPTLanguageModel.load_from_checkpoint(
        checkpoint_path="/home2/cye73/StatisticalML/nanoGPT/nanoGPT-epoch=6-val_loss=1.38.ckpt"
    )

    print(
        sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        "M trainable parameters",
    )

    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath=output_path,  # Directory to save the model
        filename=f"nanoGPT-{{epoch}}-{{val_loss:.2f}}",  # Saves the model with epoch and val_loss in the filename
        save_top_k=1,  # Number of best models to save; -1 means save all of them
        mode="min",  # 'max' means the highest max_acc will be considered as the best model
        verbose=True,  # Logs a message whenever a model checkpoint is saved
    )

    wandb_logger = WandbLogger(project=experiment_name)

    trainer = L.Trainer(
        limit_train_batches=0,
        devices=[0],
        max_epochs=0,
        # limit_train_batches=1000,
        # devices=[0, 1, 2, 3],
        # max_epochs=10,
        limit_val_batches=200,
        num_sanity_val_steps=0,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        enable_progress_bar=True,
        callbacks=[model_checkpoint],
        accumulate_grad_batches=True,
        logger=wandb_logger,
    )

    trainer.fit(model, datamodule=data_module)

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long)
    generated_text = model.generate(idx=context, max_new_tokens=1000)
    print(generated_text)
    open("more.txt", "w").write(model.generate(context, max_new_tokens=10000))


if __name__ == "__main__":
    main()
