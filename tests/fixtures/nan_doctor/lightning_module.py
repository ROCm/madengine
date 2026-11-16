import torch
import lightning as L


class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(64, 64)

    def training_step(self, batch, batch_idx):
        loss = self.net(batch).pow(2).mean()
        if torch.isnan(loss):
            raise RuntimeError("nan loss")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


if __name__ == "__main__":
    model = torch.compile(LitModel())
    trainer = L.Trainer(max_steps=100, devices=8, strategy="ddp")
    trainer.fit(model)
