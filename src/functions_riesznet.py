import torch
from torch import nn


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class RieszNet(nn.Module):
    def __init__(self, input_dim: int, dim_common: int = 256,
                 dim_heads: int = 64, drop_out: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.dim_common = dim_common
        self.dim_heads = dim_heads
        self.drop_out = drop_out

        self.common_layers = nn.Sequential(
            nn.Linear(self.input_dim, self.dim_common),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )
        self.regression_treated = nn.Sequential(
            nn.Linear(self.dim_common, self.dim_heads),
            nn.ELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim_heads, 1),
        )
        self.regression_control = nn.Sequential(
            nn.Linear(self.dim_common, self.dim_heads),
            nn.ELU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.dim_heads, 1),
        )
        self.rr = nn.Sequential(nn.Linear(self.dim_common, 1))
        self.epsi = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = self.common_layers(x)
        reg = self.regression_control(features) * (1 - x[:, [0]]) + self.regression_treated(features) * x[:, [0]]
        rr = self.rr(features)
        srr = reg + self.epsi * rr
        return torch.cat([reg, rr, srr], dim=1)


class RieszModel:
    def __init__(self, learner, device: str,  lambda1: int = 0,
                 lambda2: int = 0):
        self.device = device
        self.learner = learner.to(device)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    @staticmethod
    def _moments(fcn, x):
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
        return fcn(t1) - fcn(t0)

    def loss(self, x, y):
        riesz = lambda x: self.learner(x)[:, [1]]
        pred = self.learner(x)
        loss = torch.mean((pred[:, [0]] - y) ** 2)
        loss += self.lambda1*torch.mean(pred[:, [1]] ** 2 - 2*self._moments(riesz, x))
        loss += self.lambda2*torch.mean((pred[:, [2]] - y) ** 2)
        return loss

    def train(self, dataloader: torch.utils.data.DataLoader,
              dataloader_valid: torch.utils.data.DataLoader,
              epochs: int, lr: float = 1e-3, patience: int = 10,
              min_delta: float = 0):
        optimizer = torch.optim.Adam(params=self.learner.parameters(), lr=lr)
        early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

        for epoch in range(epochs):

            self.learner.train()

            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                loss = self.loss(X, y)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            self.learner.eval()
            loss_eval = 0
            for X, y in dataloader_valid:
                X, y = X.to(self.device), y.to(self.device)
                loss_eval += self.loss(X, y)

            if early_stopper.early_stop(loss_eval):
                print(f'Early stopping at epoch {epoch}')
                break

    def doubly_robust(self, x, y, ssr=True):
        self.learner.eval()
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            idx_y_pred = 2 if ssr else 0
            y1 = self.learner(t1)[:, [idx_y_pred]]
            y0 = self.learner(t0)[:, [idx_y_pred]]
            pred = self.learner(x)
        return y1 - y0 + pred[:, [1]]*(y-pred[:, [idx_y_pred]])

    def regression_adjustment(self, x, y):
        self.learner.eval()
        with torch.no_grad():
            t1 = torch.cat([torch.ones((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            t0 = torch.cat([torch.zeros((x.shape[0], 1)).to(device), x[:, 1:]], dim=1)
            y1 = self.learner(t1)[:, [0]]
            y0 = self.learner(t0)[:, [0]]
        return y1 - y0

    def riesz_2(self, x, y):
        self.learner.eval()
        with torch.no_grad():
            idx_y_pred = 2
            pred = self.learner(x)
        return pred[:, [1]]*(y-pred[:, [idx_y_pred]])
