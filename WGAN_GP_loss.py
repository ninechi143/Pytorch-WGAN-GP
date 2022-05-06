import torch
import torch.nn as nn
import torch.nn.functional as F


class WGAN_GP_Disc_Loss(nn.Module):

    def __init__(self , coeffient = 10. , device = None):

        super(WGAN_GP_Disc_Loss, self).__init__()
        self.Lambda = coeffient
        self.device = device

    def forward(self, real_logits, fake_logits , inter_logits , inter_images):

        gradients = torch.autograd.grad(outputs = inter_logits, inputs = inter_images,
                                        grad_outputs = torch.ones(inter_logits.size()).to(self.device),
                                        create_graph = True, retain_graph = True, only_inputs=True)[0]

        gradients = gradients.view(gradients.shape[0] , -1)
        gradient_penalty = self.Lambda * ((gradients.norm(2 , 1) - 1) ** 2).mean()

        loss = torch.mean(fake_logits) - torch.mean(real_logits) + gradient_penalty

        return loss


class WGAN_GP_Gen_Loss(nn.Module):

    def __init__(self):

        super(WGAN_GP_Gen_Loss, self).__init__()

    def forward(self, fake_logits):

        loss =  -1 * torch.mean(fake_logits)

        return loss


class Hinge_Disc_Loss(nn.Module):

    def __init__(self , coeffient = 10. , device = None):

        super(Hinge_Disc_Loss, self).__init__()
        self.Lambda = coeffient
        self.device = device

    def forward(self, real_logits, fake_logits , inter_logits , inter_images):

        G_part = torch.mean(torch.relu(1.0 + fake_logits))
        D_part = torch.mean(torch.relu(1.0 - real_logits))

        loss = G_part + D_part

        return loss