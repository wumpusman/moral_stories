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


class AbstractLightning(pl.LightningModule):
    def __init__(self, model, lr=.1):
        super().__init__()
        self.model = model
        self.set_lr(lr)
    def __str__(self):
        return "AbstractLightning"
    def set_lr(self, lr):
        self._lr = lr
    def get_lr(self):
        return self._lr
    def _define_optimizer(self):
        lr = self.get_lr()
        return torch.optim.Adam(self.model.parameters(), lr=lr)
    def configure_optimizers(self):
        return self._define_optimizer()
    def _format_results_epoch_end_str(self, epoch_end_scores, prefix):
        """formatting string text so that i don't repeatedly call this loop"""
        for key in epoch_end_scores.keys(): prefix += " {} {}".format(key, epoch_end_scores[key])
        return prefix
    def _epoch_end_collate(self, step_outputs, ignore_keys=set()):
        "calculates the mean and whatever else you'd want to calculate "
        default_vals = dict()
        for step in step_outputs:
            keys = step.keys()
            for key in keys:
                if (key in ignore_keys) == False:
                    default_vals.setdefault(key, 0)

                    default_vals[key] = default_vals[key] + step[key]
        for key in default_vals.keys():
            default_vals[key] = default_vals[key] / (float(len(step_outputs)))
        return default_vals
    def training_epoch_end(self, training_step_outputs):
        results = self._epoch_end_collate(training_step_outputs)
        #string_results = self._format_results_epoch_end_str(results, "train")
        for key in results.keys(): self.log("train_{}".format(key), results[key], on_step=False, on_epoch=True,
                                            prog_bar=False)
    def validation_epoch_end(self, validation_step_outputs):
        results = self._epoch_end_collate(validation_step_outputs)
        string_results = self._format_results_epoch_end_str(results, "val")
        for key in results.keys(): self.log("val_{}".format(key), results[key], on_step=False, on_epoch=True,
                                            prog_bar=False)

    def training_step(self, batch, batch_idx):
        pass

class ClassifyMorality(AbstractLightning):
    def __init__(self, model, lr=.1):
        super().__init__(model,lr)

    def forward(self, x):
        input_tensor = x['input_ids']
        labels = x['labels']
        return self.model(input_ids = input_tensor, labels = labels)

    def _training_step(self,batch,batchidx):
        outputs = self(batch)
        loss = outputs[0]
        return loss

    def training_step(self, batch, batchidx):
        return {"loss":self._training_step(batch,batchidx)}

    def validation_step(self,batch,batchidx):
        return {"loss":self._training_step(batch,batchidx)}

class Toy(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def forward(self, x):
        print('Hi, Michael!')
        input_tensor = x['input_ids']
        labels = x['labels']
        return self.model(input_ids = input_tensor, labels = labels)

    def training_step(self, batch, batchidx):
        outputs = self(batch)
        loss = outputs[0]
        return loss

if __name__ == '__main__':
    
    roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
    model = Toy(roberta, lr = .01)