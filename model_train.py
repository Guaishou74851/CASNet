import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import H

# classic residual block
class RB(nn.Module):
    def __init__(self, nf, bias, kz=3):
        super(RB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, kz, padding=kz // 2, bias=bias),
        )
        
    def forward(self, x):
        return x + self.body(x)

# proximal mapping network (https://github.com/cszn/DPIR)
class P(nn.Module):
    def __init__(self, in_nf, out_nf):
        super(P, self).__init__()
        bias, block, nb, scale_factor = False, RB, 2, 2
        mid_nf = [16, 32, 64, 128]

        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        up = lambda nf, scale_factor: nn.ConvTranspose2d(nf, nf, scale_factor, stride=scale_factor, bias=bias)
        down = lambda nf, scale_factor: nn.Conv2d(nf, nf, scale_factor, stride=scale_factor, bias=bias)
        
        self.down1 = nn.Sequential(conv(in_nf, mid_nf[0]), *[block(mid_nf[0], bias) for _ in range(nb)], down(mid_nf[0], scale_factor))
        self.down2 = nn.Sequential(conv(mid_nf[0], mid_nf[1]), *[block(mid_nf[1], bias) for _ in range(nb)], down(mid_nf[1], scale_factor))
        self.down3 = nn.Sequential(conv(mid_nf[1], mid_nf[2]), *[block(mid_nf[2], bias) for _ in range(nb)], down(mid_nf[2], scale_factor))
        
        self.body  = nn.Sequential(conv(mid_nf[2], mid_nf[3]), *[block(mid_nf[3], bias) for _ in range(nb)], conv(mid_nf[3], mid_nf[2]))
        
        self.up3 = nn.Sequential(up(mid_nf[2], scale_factor), *[block(mid_nf[2], bias) for _ in range(nb)], conv(mid_nf[2], mid_nf[1]))
        self.up2 = nn.Sequential(up(mid_nf[1], scale_factor), *[block(mid_nf[1], bias) for _ in range(nb)], conv(mid_nf[1], mid_nf[0]))
        self.up1 = nn.Sequential(up(mid_nf[0], scale_factor), *[block(mid_nf[0], bias) for _ in range(nb)], conv(mid_nf[0], out_nf))

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x = self.body(x3)
        x = self.up3(x + x3)  # three skip connections for the last three scales
        x = self.up2(x + x2)
        x = self.up1(x + x1)
        return x

class Phase(nn.Module):
    def __init__(self, img_nf, B):
        super(Phase, self).__init__()
        bias, nf, nb, onf = True, 8, 3, 3  # config of E

        self.rho = nn.Parameter(torch.Tensor([0.5]))
        self.P = P(img_nf + onf, img_nf)  # input: [Z | saliency_feature]
        self.B = B  # default: 32
        self.E = nn.Sequential(  # saliency feature extractor
            nn.Conv2d(1, nf, 1, bias=bias),
            *[RB(nf, bias, kz=1) for _ in range(nb)],
            nn.Conv2d(nf, onf, 1, bias=bias)
        )
    
    def forward(self, x, cs_ratio_map, PhiT_Phi, PhiT_y, mode, shape_info):
        b, l, h, w = shape_info
        
        # block gradient descent
        x = x - self.rho * (PhiT_Phi.matmul(x) - PhiT_y)
        
        # saliency information guided proximal mapping (with RTE strategy)
        x = x.reshape(b, l, -1).permute(0, 2, 1)
        x = F.fold(x, output_size=(h, w), kernel_size=self.B, stride=self.B)
        x_rotated = H(x, mode)
        cs_ratio_map_rotated = H(cs_ratio_map, mode)
        saliency_feature = self.E(cs_ratio_map_rotated)
        x_rotated = x_rotated + self.P(torch.cat([x_rotated, saliency_feature], dim=1))
        return H(x_rotated, mode, inv=True)  # inverse of H

# saliency detector
class D(nn.Module):
    def __init__(self, img_nf):
        super(D, self).__init__()
        bias, block, nb, mid_nf = False, RB, 3, 32
        conv = lambda in_nf, out_nf: nn.Conv2d(in_nf, out_nf, 3, padding=1, bias=bias)
        self.body = nn.Sequential(conv(img_nf, mid_nf), *[block(mid_nf, bias) for _ in range(nb)], conv(mid_nf, 1))
        
    def forward(self, x):
        return self.body(x).reshape(*x.shape[:2], -1).softmax(dim=2).reshape_as(x)

# error correction of BRA
def batch_correct(Q, target_sum, N):
    b, l = Q.shape
    i, max_desc_step = 0, 10
    while True:
        i += 1
        Q = torch.clamp(Q, 0, N).round()
        d = Q.sum(dim=1) - target_sum  # batch delta
        if float(d.abs().sum()) == 0.0:
            break
        elif i < max_desc_step:  # 1: uniform descent
            Q = Q - (d / l).reshape(-1, 1).expand_as(Q)
        else:  # 2: random allocation
            for j in range(b):
                D = np.random.multinomial(int(d[j].abs().ceil()), [1.0 / l] * l, size=1)
                Q[j] -= int(d[j].sign()) * torch.Tensor(D).squeeze(0).to(Q.device)
    return Q
    
class CASNet(nn.Module):
    def __init__(self, phase_num, B, img_nf, Phi_init):
        super(CASNet, self).__init__()
        self.phase_num = phase_num
        self.phase_num_minus_1 = phase_num - 1
        self.B = B
        self.N = B * B
        self.Phi = nn.Parameter(Phi_init.reshape(self.N, self.N))
        self.RS = nn.ModuleList([Phase(img_nf, B) for _ in range(phase_num)])
        self.D = D(img_nf)
        self.index_mask = torch.arange(1, self.N + 1)
        self.epsilon = 1e-6

    def forward(self, x, q, modes):
        b, c, h, w = x.shape

        # saliency detection
        S = self.D(x)  # saliency map

        # CS ratio allocation (with BRA method)
        x_unfold = F.unfold(x, kernel_size=self.B, stride=self.B).permute(0, 2, 1)  # shape: (b, l, img_nf * B * B)
        l = x_unfold.shape[1]  # block number of an image patch
        
        Q = q * l * S  # measurement size map
        Q_unfold = F.unfold(Q, kernel_size=self.B, stride=self.B).permute(0, 2, 1).sum(dim=2)  # sumpooling, shape: (b, l)
        Q_unfold = batch_correct(Q_unfold, q * l, self.N) + self.epsilon * (Q_unfold - Q_unfold.detach())  # error correction
        
        # divide image patch into blocks
        block_stack = x_unfold.reshape(-1, c * self.N, 1)  # shape: (b * l, img_nf * B * B, 1)
        block_volume = block_stack.shape[1]  # img_nf * B * B

        # generate sampling matrices
        L = block_stack.shape[0]  # total block number of batch
        Phi_stack = self.Phi.unsqueeze(0).repeat(L, 1, 1)
        
        index_mask = self.index_mask.unsqueeze(0).repeat(L, 1).to(Phi_stack.device)
        q_stack = Q_unfold.reshape(-1, 1).repeat(1, Phi_stack.shape[1])
        cur_mask = F.relu(q_stack - index_mask + 1.0).sign() + self.epsilon * (q_stack - q_stack.detach())
        Phi_stack = Phi_stack * cur_mask.unsqueeze(2)

        # sample and initialize respectively
        # y = Phi_stack.matmul(block_stack)  # sampling, shape: (b * l, N, 1)
        # PhiT_y = Phi_stack.permute(0, 2, 1).matmul(y)  # initialization, shape: (b * l, img_nf * B * B, 1)

        # sample and initialize simultaneously
        PhiT_Phi = Phi_stack.permute(0, 2, 1).matmul(Phi_stack)
        PhiT_y = PhiT_Phi.matmul(block_stack)
        
        x = PhiT_y

        # get expanded CS ratio map R'
        cs_ratio_map = (Q_unfold.detach() / self.N).unsqueeze(2).repeat(1, 1, block_volume).permute(0, 2, 1)
        cs_ratio_map = F.fold(cs_ratio_map, output_size=(h, w), kernel_size=self.B, stride=self.B)

        # recover step-by-step
        shape_info = [b, l, h, w]
        for i in range(self.phase_num):
            x = self.RS[i](x, cs_ratio_map, PhiT_Phi, PhiT_y, modes[i], shape_info)
            if i < self.phase_num_minus_1:
                x = F.unfold(x, kernel_size=self.B, stride=self.B).permute(0, 2, 1)
                x = x.reshape(L, -1, 1)
        return x
