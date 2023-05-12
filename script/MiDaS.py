import torch

class MiDaS(torch.nn.Module):

  def __init__(self, model_type, max_distance = 1000):
    super(MiDaS, self).__init__()

    #self.mean = torch.tensor([0.485, 0.456, 0.406])
    #self.std = torch.tensor([0.229, 0.224, 0.225])
    self.midas = torch.hub.load('intel-isl/MiDaS', model_type)
    self.max_distance = max_distance

  def forward(self, x):
    x = self.midas(x)
    x = torch.clamp(x, 0, self.max_distance) / self.max_distance * 255.0
    x = x.unsqueeze(0)
    return x
