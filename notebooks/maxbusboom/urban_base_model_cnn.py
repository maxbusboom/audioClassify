import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

class UrbanSoundBase(nn.Module):
  def training_step(self, batch_images):
    images,labels = batch_images
    outputs = self(images)
    loss = F.cross_entropy(outputs, labels)
    acc = self.accuracy(outputs, labels)
    return {'loss' : loss, 'acc' : acc}

  def accuracy(self,outputs, labels, logits=True):
    output_softmaxed = F.softmax(outputs, dim=1)
    vals,predictions, = torch.max(output_softmaxed, dim=1)
    assert predictions.shape == labels.shape
    return torch.tensor(torch.sum(predictions == labels).item()/outputs.size(0))

  def validation_step(self, batch_images, device):
    images,labels = batch_images
    images = images.to(device)
    labels = labels.to(device)
    outputs = self(images)
    loss = F.cross_entropy(outputs, labels)
    acc = self.accuracy(outputs, labels)

    return {'val_loss' : loss, 'val_acc' : acc}

  def epoch_end(self, history):
    loss = torch.stack([batch['val_loss'] for batch in history]).mean().item()
    accuracy = torch.stack([batch['val_acc'] for batch in history]).mean().item()

    return {'val_loss': loss, 'val_acc' : accuracy}

  @torch.no_grad()
  def evaluate_validation(self, valid_dataloader, device):
    return [self.validation_step(image_label_batch, device) for image_label_batch in valid_dataloader]