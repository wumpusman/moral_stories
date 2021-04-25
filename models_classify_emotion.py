
from transformers import RobertaTokenizer, RobertaModel
import pytorch_lightning as pl
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support as f1
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import callbacks
from pytorch_lightning import Trainer
from importlib import reload
from transformers import pipeline
from sklearn.metrics import accuracy_score
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support as f1
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append("../../")
try:
    import dataloader_yelp
except:
    from core import dataloader_yelp

##
#
#torch.set_printoptions(sci_mode=False)

def load_model(type_emotion_classify_model,original_roberta_model, checkpoint_path, use_cuda=True):
    #"/home/experiments/biasedtext/RobertaWeightedProbHead-_epoch=8--f1:val_f1=0.90--trloss=0.30.ckpt"

    internal_pytorch_model = type_emotion_classify_model(original_roberta_model)
    lightning_model = RobertaPolarHead(internal_pytorch_model)
    if use_cuda == False:
        checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
    lightning_model.load_state_dict(checkpoint["state_dict"])
    return lightning_model

class RobertaPolarHead(pl.LightningModule):

    def __init__(self, original_roberta_wrapper, train_head_only=True):
        super().__init__()
        self.modified_roberta = original_roberta_wrapper
        self.train_head_only = train_head_only

    def toggle_train_headonly(self, train_head_only=True):
        self.train_head_only = train_head_only

    def forward(self, x):
        input_tensor = x[0]
        labels = x[1]
        mask = x[2]
        return self.modified_roberta([input_tensor, labels, mask])

    def __str__(self):
        return str(self.modified_roberta)

    def configure_optimizers(self):
        # if self.train_head_only:
        #    return torch.optim.Adam(self.modified_roberta.relevant_head.parameters(),lr=.001)
        # else:
        #    return torch.optim.AdamW(self.modified_roberta.parameters(),lr=.001)

        optimzer = torch.optim.Adam(self.modified_roberta.relevant_head.parameters(), lr=.1)

        if self.train_head_only == False:
            optimzer = torch.optim.Adam([{"params": self.modified_roberta.relevant_head.parameters()},
                                         {"params": self.modified_roberta.get_roberta().parameters()}],
                                        lr=.0001)

        scheduler = ReduceLROnPlateau(optimzer, mode="min", patience=3,
                                      factor=.3, )

        # scheduler = lr_scheduler.OneCycleLR(optimzer,max_lr=.1,epochs=10,steps_per_epoch=10)

        scheduler = lr_scheduler.CosineAnnealingLR(optimzer, T_max=10, eta_min=.0003)
        scheduler_info = {"lr_scheduler": scheduler, "interval": 'epoch', "frequency": 1
            , "strict": True, "monitor": "loss"}  # strict means if it fails, output something
        # ReduceLROnPlateau()

        scheduler_info.update({"optimizer": optimzer})

        return scheduler_info

    def training_step(self, batch, batch_idx):

        labels = batch[1]
        mean_results, org_results = self(batch)
        results = mean_results

        loss_obj = nn.NLLLoss()

        loss = loss_obj(results, labels)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        labels = batch[1]
        predictions = None
        with torch.no_grad():
            predictions, org_predictions = self(batch)

        predicted_class = torch.argmax(predictions, dim=-1)
        precision, recall, f1_score, support = f1(labels.cpu(), predicted_class.cpu(), average="macro", zero_division=1)

        # self.log("val_f1",f1_score,on_step=True,on_epoch=True,prog_bar=False)
        return {"val_f1": f1_score, "val_precision": precision, "val_recall": precision}

    def training_epoch_end(self, training_step_outputs):
        total_loss = 0

        for i in training_step_outputs:
            total_loss += i["loss"]

        print("loss {}".format(total_loss / (float(len(training_step_outputs)))))

    def validation_epoch_end(self, validation_step_outputs):
        f1_total = 0
        precision = 0
        for i in validation_step_outputs:
            f1_total += i["val_f1"]
            precision += i["val_precision"]

        f1_score = f1_total / float(len(validation_step_outputs))
        precision = precision / float(len(validation_step_outputs))
        print("f1 score {}".format(f1_score))
        self.log("val_f1", f1_score, on_step=False, on_epoch=True, prog_bar=False)
        # note alt form is to return log {"log":{"val_f1":f1_score}}
        # print(precision)

    def test_step(self, batch, batch_idx):

        labels = batch[1]
        predictions = None
        with torch.no_grad():
            predictions, org_predictions = self(batch)

        predicted_class = torch.argmax(predictions, dim=-1)
        precision, recall, f1_score, support = f1(labels.cpu(), predicted_class.cpu(), average="macro", zero_division=1)
        accuracy = accuracy_score(labels.cpu(), predicted_class.cpu())
        return {"val_f1": f1_score, "val_precision": precision, "val_recall": precision, "accuracy": accuracy}

    def test_epoch_end(self, validation_step_outputs):

        f1_total = 0
        precision = 0
        accuracy = 0
        for i in validation_step_outputs:
            f1_total += i["val_f1"]
            precision += i["val_precision"]
            accuracy += i["accuracy"]

        f1_score = f1_total / float(len(validation_step_outputs))
        precision = precision / float(len(validation_step_outputs))
        accuracy = accuracy / float(len(validation_step_outputs))
        print(" f1_score {}".format(f1_score))
        print("accuracy {}".format(accuracy))
        # self.write("test_f1",f1_score)



class RobertaPolarHeadTorch(nn.Module):
    def __init__(self, original_roberta):
        super().__init__()
        self.original_robert = [original_roberta]  # so don't have to deal with paraemters being saved for roberta

        self.relevant_head = self._define_relevant_head()

    def _define_relevant_head(self):
        self.linear = nn.Sequential(nn.Linear(768, 2), nn.LogSoftmax(dim=-1))

        self.do_output_original = False

        return nn.ModuleList([self.linear])

    def __str__(self):
        return "RobertaPolarHead"

    def get_roberta(self):
        return self.original_robert[0]

    def forward(self, x):
        input_tensor = x[0]
        labels = x[1]
        mask = x[2]

        # print(input_tensor)
        output_org = self.original_robert[0](input_tensor, attention_mask=mask).last_hidden_state

        output = self.linear(output_org)

        return output.mean(dim=1), output


class RobertaProbHeadTorch(RobertaPolarHeadTorch):
    def __init__(self, original_roberta):
        super().__init__(original_roberta)

        self.relevant_head = self._define_relevant_head()

    def _define_relevant_head(self):
        self.linear = nn.Sequential(nn.Linear(768, 2))
        self.confidence = nn.Sequential(nn.Linear(768, 1), nn.LayerNorm((300, 1)))
        self.log_soft = nn.LogSoftmax(dim=1)

        relevant_head = nn.ModuleList([self.linear, self.confidence, self.log_soft])
        return relevant_head

    def __str__(self):
        return "RobertaWeightedProbHead"

    def forward(self, x):
        input_tensor = x[0]
        labels = x[1]
        mask = x[2]

        # print(input_tensor)
        output_org = self.original_robert[0](input_tensor, attention_mask=mask).last_hidden_state

        output = self.linear(output_org)

        probs = self.confidence(output_org)
        values = self.linear(output_org)

        weighted_prediction = self.log_soft((probs * values).mean(dim=1))

        return weighted_prediction, probs * values