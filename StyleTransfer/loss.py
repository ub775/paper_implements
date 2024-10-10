# content loss
# vgg19 feature map -> deep image representation
# MSE

# style loss
# gram matrix -> function
# MSE

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        # MSE loss
        loss = F.mse_loss(x, y)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def gram_matrix(self, x:torch.Tensor):
        b, c, h, w = x.size()
        # reshape
        features = x.view(b, c, h * w)
        features_T = features.transpose(1, 2)
        G = torch.matmul(features, features_T)
        return G.div(b*c*h*w)

    def forward(self, x:torch.Tensor, y:torch.Tensor):
        # MSE loss
        Gx = self.gram_matrix(x)
        Gy = self.gram_matrix(y)
        loss = F.mse_loss(Gx, Gy)
        return loss