import torch

class MiDaS(torch.nn.Module):

  def __init__(self, model_type):
    super(Model, self).__init__()

    self.mean = torch.tensor([0.485, 0.456, 0.406])
    self.std = torch.tensor([0.229, 0.224, 0.225])
    self.midas = torch.hub.load('intel-isl/MiDaS', model_type)

  def forward(self, x):
    x = x / 255.0
    x = (x - 0.45) / 0.225
    x = self.midas(x)
    x = x - torch.min(x)
    x = x / torch.max(x)
    x = x.unsqueeze(0)
    return x
