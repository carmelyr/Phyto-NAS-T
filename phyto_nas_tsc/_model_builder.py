import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

def build_model(model_type, **kwargs):
    print(f"Building model type: {model_type}")
    input_size = kwargs.get("input_size", 1)
    output_size = kwargs.get("output_size", 2)

    if model_type == "FCNN":
        return FCNN(
            input_size=input_size,
            hidden_units=kwargs.get("hidden_units", 64),
            output_size=output_size,
            num_layers=kwargs.get("num_layers", 8),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            learning_rate=kwargs.get("learning_rate", 1e-3)
        )
    elif model_type == "CNN":
        return CNN(
            input_channels=kwargs.get("input_channels", 1),
            num_filters=kwargs.get("num_filters", 32),
            kernel_size=kwargs.get("kernel_size", 3),
            output_size=output_size,
            dropout_rate=kwargs.get("dropout_rate", 0.3),
            learning_rate=kwargs.get("learning_rate", 1e-4)
        )
    elif model_type == "GRU":
        return GRU(
            input_size=input_size,
            hidden_units=kwargs.get("hidden_units", 128),
            output_size=output_size,
            num_layers=kwargs.get("num_layers", 5),
            dropout_rate=kwargs.get("dropout_rate", 0.4),
            bidirectional=kwargs.get("bidirectional", True),
            learning_rate=kwargs.get("learning_rate", 1e-3)
        )
    elif model_type == "LSTM":
        return LSTM(
            input_size=input_size,
            hidden_units=kwargs.get("hidden_units", 128),
            output_size=output_size,
            num_layers=kwargs.get("num_layers", 2),
            dropout_rate=kwargs.get("dropout_rate", 0.3),
            bidirectional=kwargs.get("bidirectional", True),
            attention=kwargs.get("attention", True),
            learning_rate=kwargs.get("learning_rate", 1e-3),
            weight_decay=kwargs.get("weight_decay", 0)
        )
    elif model_type == "Transformer":
        return TransformerModel(
            input_dim=input_size,
            num_heads=kwargs.get("num_heads", 4),
            num_layers=kwargs.get("num_layers", 2),
            hidden_dim=kwargs.get("hidden_dim", 64),
            output_size=output_size,
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            learning_rate=kwargs.get("learning_rate", 1e-4),
            weight_decay=kwargs.get("weight_decay", 1e-4)
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")


# Fully Connected Neural Network (FCNN)
class FCNN(pl.LightningModule):
    def __init__(self, input_size, hidden_units=64, output_size=2, num_layers=8, dropout_rate=0.2, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Linear(input_size, hidden_units))
        self.layers.append(nn.BatchNorm1d(hidden_units))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(p=dropout_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(nn.BatchNorm1d(hidden_units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_units, output_size))
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Convolutional Neural Network (CNN)
class CNN(pl.LightningModule):
    def __init__(self, input_channels=1, num_filters=32, kernel_size=3, output_size=2, 
                 dropout_rate=0.3, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, num_filters, kernel_size, padding='same')
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.conv2 = nn.Conv1d(num_filters, num_filters*2, kernel_size, padding='same')
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.conv3 = nn.Conv1d(num_filters*2, num_filters*4, kernel_size, padding='same')
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.conv4 = nn.Conv1d(num_filters*4, num_filters*8, kernel_size, padding='same')
        self.bn4 = nn.BatchNorm1d(num_filters*8)
        
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(num_filters*8, output_size)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add channel dimension if missing
            
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# LSTM-based Model (not a standard model, but optimized model (Phase 2 of Phyto-NAS-TSC process is done on this model))
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


# GRU-based Model
class GRU(pl.LightningModule):
    def __init__(self, input_size, hidden_units=128, output_size=2, num_layers=5, 
                 dropout_rate=0.4, bidirectional=True, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_units,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_units * (2 if bidirectional else 1), output_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension if missing
            
        gru_out, _ = self.gru(x)
        gru_out = gru_out[:, -1, :]  # Take last time step
        return self.fc(gru_out)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

# Transformer-based Model
class TransformerModel(pl.LightningModule):
    def __init__(self, input_dim, num_heads=4, num_layers=2, hidden_dim=64, 
                 output_size=2, dropout_rate=0.2, learning_rate=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = num_heads * (hidden_dim // num_heads + 1)
        
        # Input embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_size)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Mean pooling over time dimension
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.train_acc(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if y.dim() > 1:
            y = torch.argmax(y, dim=1)
            
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        self.val_acc(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x