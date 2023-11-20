import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import F1Score


class Classifier(pl.LightningModule):
    def __init__(
            self,
            model,
            num_classes,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model = model
        self.num_classes = num_classes

        self.steps_current_epoch_outputs = []
        self.ce_losses_current_epoch_outputs = []
        self.f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def training_step(self, batch, _):
        return self._step(batch)

    def test_step(self, batch, *args, **kwargs):
        return self._step(batch)

    def on_test_epoch_end(self):
        f1_macro_epoch_mean = torch.stack(self.steps_current_epoch_outputs).mean()

        self.log('testing_f1_macro_epoch_mean', f1_macro_epoch_mean)
        self.steps_current_epoch_outputs.clear()

    def on_train_epoch_end(self):
        f1_macro_epoch_mean = torch.stack(self.steps_current_epoch_outputs).mean()
        ce_epoch_mean = torch.stack(self.ce_losses_current_epoch_outputs).mean()

        self.log('training_f1_macro_epoch_mean', f1_macro_epoch_mean)
        self.log('training_cross_entropy_epoch_mean', ce_epoch_mean)

        self.steps_current_epoch_outputs.clear()
        self.ce_losses_current_epoch_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())

    def _step(self, batch):
        x, y = batch
        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)

        self.f1(logits, y)
        f1 = self._compute_f1_macro()
        self.steps_current_epoch_outputs.append(f1)
        self.ce_losses_current_epoch_outputs.append(loss)

        return loss

    def _compute_f1_macro(self):
        f1_score = self.f1.compute()
        self.f1.reset()
        return f1_score
