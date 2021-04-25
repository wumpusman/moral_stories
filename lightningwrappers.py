import torch
import pytorch_lightning as pl
from torch.optim import lr_scheduler
from torch import nn
import numpy as np
class AbstractLightning(pl.LightningModule):
    def __init__(self, model, lr=.1, scheduler_str="reducelr", scheduler_params={}):
        """
        scheduler_str: 'reducelr','cosine','onecycle'
        """
        super().__init__()
        self.model = model

        self.set_lr(lr)
        self._scheduler_type = scheduler_str
        self._scheduler_params = scheduler_params

    def __str__(self):
        return "AbstractLightning"

    def set_lr(self, lr):
        self._lr = lr

    def get_lr(self):
        return self._lr

    def get_scheduler_params(self):
        return self._scheduler_type, self._scheduler_params

    def set_scheduler(self, scheduler_str, scheduler_params):
        self._scheduler_type = scheduler_str
        self._scheduler_params = scheduler_params

    def _define_optimizer(self):
        lr = self.get_lr()
        return torch.optim.Adam(self.model.parameters(), lr=lr)

    def configure_optimizers(self):
        lr = self.get_lr()
        scheduler_name, scheduler_params = self.get_scheduler_params()

        optimizer = self._define_optimizer()

        scheduler_choice = self._select_scheduler()
        scheduler_type = scheduler_choice[0]
        scheduler_params = scheduler_choice[1]
        scheduler_params.update({"optimizer": optimizer})

        scheduler = scheduler_type(**scheduler_params)

        scheduler_info = {"lr_scheduler": scheduler, "interval": 'epoch', "frequency": 1,
                          "strict": True, "monitor": "loss", "optimizer": optimizer}

        return scheduler_info

    def _select_scheduler(self):
        lr = self.get_lr()
        scheduler_type_str, schedule_params = self.get_scheduler_params()

        schedulers = {
            "cosine": [lr_scheduler.CosineAnnealingLR,
                       {"T_max": 5, "eta_min": .0003}],

            "reducelr": [lr_scheduler.ReduceLROnPlateau,
                         {"mode": "min", "patience": 1, "factor": .3}],

            "onecycle": [lr_scheduler.OneCycleLR,
                         {"max_lr": lr, "epochs": 10, "steps_per_epoch": 10}]
        }

        relevant_scheduler = schedulers[scheduler_type_str]

        relevant_scheduler[1].update(schedule_params)

        return relevant_scheduler

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
        string_results = self._format_results_epoch_end_str(results, "train")
        print(string_results)

    def validation_epoch_end(self, validation_step_outputs):
        results = self._epoch_end_collate(validation_step_outputs)
        string_results = self._format_results_epoch_end_str(results, "val")
        for key in results.keys(): self.log("val_{}".format(key), results[key], on_step=False, on_epoch=True,
                                            prog_bar=False)
        print(string_results)

    def training_step(self, batch, batch_idx):
        pass


class ClippyLightning(AbstractLightning):
    def __init__(self, model, normalize = False,lr=.1, scheduler_str="reducelr", scheduler_params={}):
        """
        scheduler_str: 'reducelr','cosine','onecycle'
        """
        super().__init__(model, lr, scheduler_str, scheduler_params)
        self._logit_scaling =  nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        self._should_normalize = normalize

    def normalize_result(self,should_normalize):
        self._should_normalize = should_normalize
    def forward(self, x):

        original_img = x["img"]
        original_img_embed = self.model.encode_image(original_img)

        if self._should_normalize == True: #rescales and norms
            original_img_embed = original_img_embed/original_img_embed.norm(dim=-1,keepdim=True)
            original_img_embed = ((self._logit_scaling * (original_img_embed @ original_img_embed.T)).softmax(dim=-1))

        return original_img_embed

    def training_step(self, batch, batch_idx):
        pass

    def _validation_step(self, batch, batch_idx):
        pass
