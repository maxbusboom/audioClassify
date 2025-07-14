import torch
import torch.nn as nn
import torch.nn.functional as F

from urban_base_model_cnn import UrbanSoundBase

class CNNModel_6Layers(UrbanSoundBase):
    """
    Model : 4 CNN Layers, 2NN Layers
    Layer Size : 16, 32, 64, 64, _256, _10
    Drop Outs : False, True, False, True, False
    Drop Values : _, 0.3, _, 0.3, _,
    Max Pool : False, True, False, True
    Max Pool Values : _, 2, _, 2
    """
    def __init__(self, output_size, sample_row_size, sample_col_size, sample_channels=1, sample_batch_size=64):
      super().__init__()
      self.flatten_size = None
      self.cnn_block = nn.Sequential(
          nn.Conv2d(1, 16, 3),
          nn.ReLU(),
          nn.Conv2d(16, 32, 3),
          nn.MaxPool2d(2,2),
          nn.ReLU(),
          nn.Dropout(0.3),


          nn.Conv2d(32, 64, 3),
          nn.ReLU(),
          nn.Conv2d(64, 64, 3),
          nn.MaxPool2d(2,2),
          nn.ReLU(),
          nn.Dropout(0.3)
          )
      
      if self.flatten_size is None:
        random_input = torch.randn((sample_batch_size, sample_channels, sample_row_size, sample_col_size))
        self.flatten_size = list(self.cnn_block(random_input).reshape(random_input.shape[0], -1).shape)[1]
      
      self.linear_block = nn.Sequential(
          nn.Flatten(),
          nn.Linear(self.flatten_size, 256),
          nn.ReLU(),
          nn.Linear(256, output_size),
      )

    def forward(self, xb):
      out = self.cnn_block(xb)
      out = self.linear_block(out)
      return out