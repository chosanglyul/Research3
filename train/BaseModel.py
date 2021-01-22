import torch
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    def __init__(self, loss_fn, metrics_fn=[], optim=torch.optim.Adam, lr=1e-3):
        super().__init__()
        self.metrics_fn = metrics_fn
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr
    
    def forward(self, X):
        return X
        
    def configure_optimizers(self):
        return self.optim(self.parameters(), lr=self.lr)
    
    def forward_batch(self, batch):
        X, y = batch
        return self.forward(X), y
    
    def log_metrics(self, h, y, step_name):
        metrics = {'{}/{}'.format(name, step_name) : fn(h, y) for fn, name in self.metrics_fn}
        metrics['loss/{}'.format(step_name)] = self.loss_fn(h, y)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics['loss/{}'.format(step_name)]
    
    def training_step(self, train_batch, batch_idx):
        h, y = self.forward_batch(train_batch)
        return self.log_metrics(h, y, 'train')

    def validation_step(self, valid_batch, batch_idx):
        h, y = self.forward_batch(valid_batch)
        self.log_metrics(h, y, 'valid')
    
    def test_step(self, test_batch, batch_idx):
        h, y = self.forward_batch(test_batch)
        self.log_metrics(h, y, 'test')
        return h, y