import torch
import torch.nn as nn
import torch.nn.functional as F

from urban_base_model_cnn import UrbanSoundBase

class DynamicModel(UrbanSoundBase):
  def __init__(self, input_channels, cnn_hidden_channels, hidden_sizes, output_size, sample_image_batch_dim):
    """
    input_chanels (int) : no of channels in incoming input, 2D has 1 channel 3D has 3.
    cnn_hidden_channels (list(int)) : list of channels for series of CNN layers.
    hidden_size (list(int)) : list of hidden neurons for linear layers.
    output_size (int) : No of neurons in last layer (10 if classifying for 10 classes)
    sample_image_batch_dim (tuple(int,int,int,int)) : 4D tuple specifying (batch_size, num_channels, size_a, size_b)
    """

    """
    max_pool layer is added after every 2 cnn layers.
    For Sample Model Run:
    input = torch.randn(32,1,40,174)
    input_channels = 1
    cnn_hidden_channels = [16, 32, 32, 64]
    hidden_sizes = [128]
    output_size = 10
    sample_image_batch_dim = (32,1,40,174)
    model = DynamicModel(input_channels, cnn_hidden_channels, hidden_sizes, output_size, sample_image_batch_dim)
    """
    super().__init__()
    self.flatten_size = None
    self.input_dim = input_channels
    self.cnn_hidden_dims = cnn_hidden_channels
    self.hidden_dims = hidden_sizes
    self.output_dim = output_size
    self.layers = nn.ModuleList()

    current_size = self.input_dim

    for idx,hidden_dim in enumerate(self.cnn_hidden_dims):
      if idx % 2:
        self.layers.append(self.cnn_block(current_size, hidden_dim, max_pool=False))
      else:
        self.layers.append(self.cnn_block(current_size, hidden_dim, max_pool=True))
      current_size = hidden_dim

    if self.flatten_size is None:
      random_input = torch.randn(sample_image_batch_dim)
      for layer in self.layers:
        random_input = layer(random_input)

      self.flatten_size = list(random_input.reshape(random_input.shape[0], -1).shape)[1]


    self.layers.append(nn.Flatten()) #To Flatten the CNN Output

    current_size = self.flatten_size

    for hidden_dim in self.hidden_dims:
      self.layers.append(self.nn_block(current_size, hidden_dim))
      current_size = hidden_dim

    self.layers.append(self.nn_block(current_size, output_size, relu=False))
    
  def forward(self, x):
    out = x
    for layer in self.layers:
      out = layer(out)
    return out

  def cnn_block(self,input_size, output_size, kernel_size = 3, max_pool_size = 2, max_pool=True, relu = True):
    """
    Creates a custom cnn_block
    """
    cnn_custom_block = nn.Sequential()
    cnn_custom_block.add_module("conv_layer" , nn.Conv2d(input_size, output_size, kernel_size))

    if max_pool:
      cnn_custom_block.add_module("max_pool_layer", nn.MaxPool2d(max_pool_size))

    if relu:
      cnn_custom_block.add_module("relu_layer", nn.ReLU())

    return cnn_custom_block

  def nn_block(self, input_size, output_size, relu=True):
    """
    Creates a custom ann_block
    """

    ann_custom_block = nn.Sequential()
    ann_custom_block.add_module("linear_layer", nn.Linear(input_size, output_size))
    if relu:
      ann_custom_block.add_module("relu_layer", nn.ReLU())
    return ann_custom_block