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
import os

import Refactored_Baseline
import moral_dataset

roberta = RobertaForSequenceClassification.from_pretrained('roberta-base')
model = Refactored_Baseline.ClassifyMorality(roberta)
tokenizer = roberta = RobertaTokenizer.from_pretrained('roberta-base')

path = "/Users/garbar/Downloads/moral_stories_datasets/classification/action+context/"
dataset_name = "lexical_bias"
tasktype = moral_dataset.TaskTypes.ACTION_CONTEXT_CLS
modeltype = moral_dataset.ModelNames.ROBERTA

morality_classify_dataset = moral_dataset.MoralStoryDataLoader(os.path.join(path,dataset_name),modeltype,tasktype,
                                                               tokenizer,amount_to_process=100,batchsize=4,num_workers=3)

epochs = 3
trainer = pl.Trainer(logger = None, gpus = 0,min_epochs = 1, max_epochs = epochs,
        log_every_n_steps = 5,check_val_every_n_epoch=2,
        limit_test_batches=50,limit_train_batches=.99,limit_val_batches=.25,checkpoint_callback=False)

trainer.fit(model,morality_classify_dataset)
