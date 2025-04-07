import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

# This function builds the LSTM model based on the provided parameters.
def build_model(model_type, **kwargs):
    print(f"Building model type: {model_type}")

    if model_type != "LSTM":
        raise ValueError("Only LSTM model type is supported")

    return LSTM(
            input_size=kwargs.get("input_size", 1),
            hidden_units=kwargs.get("hidden_units", 128),
            output_size=kwargs.get("output_size", 2),
            num_layers=kwargs.get("num_layers", 2)
        )


# LSTM-based Model
class LSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_units=128, output_size=2, num_layers=2, dropout_rate=0.3, bidirectional=True, attention=True, learning_rate=1e-3, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate                  # initial learning rate
        self.weight_decay = weight_decay                    # weight decay for L2 regularization
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0   # dropout is applied only if num_layers > 1
        )

        self.ln = nn.LayerNorm(hidden_units * (2 if bidirectional else 1))      # layer normalization

        # Attention mechanism
        self.attention = None
        if attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_units * (2 if bidirectional else 1), hidden_units),
                nn.Tanh(),
                nn.Linear(hidden_units, 1, bias=False)
            )
        
        # Fully connected layers
        # first layer is a linear layer that maps the LSTM output to the hidden units
        # second layer is a linear layer that maps the hidden units to the output size
        # batch normalization is applied after the first linear layer
        # ReLU activation is applied after the first linear layer
        # dropout is applied after the first linear layer
        # output layer uses softmax activation to produce class probabilities
        self.classifier = nn.Sequential(
            nn.Linear(hidden_units * (2 if bidirectional else 1), hidden_units),
            nn.BatchNorm1d(hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_units, output_size)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()        # categorical cross-entropy loss function
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        if self.attention is not None:
            attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            context = lstm_out[:, -1, :]
            
        return self.classifier(context)

    def training_step(self, batch, batch_idx):
        x, y = batch
        
        # converts one-hot encoded labels to class indices
        # if y is one-hot encoded, convert to class indices
        # expected shape of y: (batch_size, num_classes)
        if y.dim() > 1 and y.size(1) > 1:
            y = torch.argmax(y, dim=1)
        
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        l2_lambda = 0.001                                           # L2 regularization parameter
        l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())  # L2 norm of the parameters
        loss = loss + l2_lambda * l2_norm                           # adds L2 regularization to the loss
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        x, y = batch
       
        if y.dim() > 1 and y.size(1) > 1:
            y = torch.argmax(y, dim=1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        # cyclical learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate/10,
                max_lr=self.learning_rate,
                step_size_up=200,
                cycle_momentum=False
            ),
            'interval': 'step'
        }
        return [optimizer], [scheduler]
