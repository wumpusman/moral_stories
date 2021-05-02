from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)
import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch import nn
import numpy as np

class Toy(pl.LightningModule)
    def __init__(self, model, lr):
        super.__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def forward(self, x):
        input_tensor = x[0]
        labels = x[0]
        return self.model([input_tensor, labels])

    def training_step(self, batch, batchidx)
        outputs = self(batch)
        loss = outputs[0]
        return loss

if __name__ == '__main__':
    
    roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
    model = Toy(roberta, lr = .01)