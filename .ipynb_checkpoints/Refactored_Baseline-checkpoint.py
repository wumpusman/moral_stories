from transformers import (RobertaConfig,
                          RobertaForSequenceClassification,
                          RobertaTokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup)
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class AbstractLightning(pl.LightningModule):
    def __init__(self, model, lr=.1):
        """
        scheduler_str: 'reducelr','cosine','onecycle'
        """
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

class Baseline_Model(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def forward(self, x):
        input_tensor = x['input_ids']
        labels = x['labels']
        return self.model(input_ids = input_tensor, labels = labels)

    def training_step(self, batch, batchidx):
        outputs = self(batch)
        return {'loss': outputs[0], 'preds': outputs[1], 'target': batch['labels'], 'text':''}
    
    def training_epoch_end(self, training_step_outputs):
        super().training_epoch_end(training_step_outputs)
        
        #https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
        self._metrics('Train', training_step_outputs)
        
    def validation_epoch_end(self, validation_step_outputs):
        super().validation_epoch_end(validation_step_outputs)
        
        #https://stackoverflow.com/questions/65498782/how-to-dump-confusion-matrix-using-tensorboard-logger-in-pytorch-lightning
        self._metrics('Validation', validation_step_outputs)
        
    def _metrics(self, title: str, outputs):
        #Collect items
        preds = torch.cat([tmp['preds'] for tmp in outputs])
        targets = torch.cat([tmp['target'] for tmp in outputs])
        losses = torch.cat([tmp['loss'] for tmp in outputs])
        
        #Confusion Matrix
        cm = pl.metrics.functional.confusion_matrix(preds, targets, num_classes=2)
        df_cm = pd.DataFrame(cm.numpy(), columns=range(2))
        plt.figure(figsize = (10,7))
        fig_ = sns.heatmap(df_cm, annot=True, cmap='Spectral').get_figure()
        plt.close(fig_)
        self.logger.experiment.add_figure("Confusion Matrix {}".format(title), fig_, self.current_epoch)
        
if __name__ == '__main__':
    
    roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
    
    model = Baseline_Model(roberta, lr = .01)