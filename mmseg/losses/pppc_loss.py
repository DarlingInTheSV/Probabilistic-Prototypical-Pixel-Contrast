import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from ..builder import LOSSES
from ..utils.one_hot import label_onehot
import mmcv



@LOSSES.register_module()
class PPPC_Loss(nn.Module):
    # For single GPU users
    def __init__(self, num_queries, num_negatives, temp=100, mean=False, strong_threshold=0.97, scale_factor=1,
                 domain='mix'):
        super(PPPC_Loss, self).__init__()
        self.domain = domain
        assert self.domain in ['src', 'tgt', 'mix']
        self.temp = temp
        self.mean = mean
        self.num_queries = num_queries
        self.num_negatives = num_negatives
        self.strong_threshold = strong_threshold
        self.scale_factor = scale_factor

    def uniform_loss(self, x, max_samples=16384, t=2):  #
        if len(x) ** 2 > max_samples:
            # prevent CUDA error: https://github.com/pytorch/pytorch/issues/22313
            indices = np.random.choice(len(x), int(np.sqrt(max_samples)))
            x = x[indices]
        return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

    def kl_divergence(self, mu, sigma):
        return -0.5 * (1 + sigma - mu.pow(2) - sigma.exp()).sum()
    def forward(self, mu_and_sigma, label, **kwargs):

        # prob = kwargs['raw_logits']
        bank = kwargs['bank']  # c x d
        proto_mu = bank.get_proto()[0]
        proto_sigma = bank.get_proto()[1]
        self.scale_factor = kwargs['scale']
        self.domain = kwargs['domain']
        log_vars = kwargs['log_vars']
        # weak_threshold = kwargs['thr']
        # prob = prob.detach()
        # prob = torch.softmax(prob, dim=1)
        mu = mu_and_sigma[0]
        sigma = mu_and_sigma[1]
        label = label.clone()
        label = F.interpolate(label.float(), size=mu.shape[2:], mode='nearest').long()

        class_num = 19  # 19
        batch_size, num_feat, mu_w, mu_h = mu.shape
        mu = mu.permute(0, 2, 3, 1).contiguous().view(-1, num_feat)
        sigma = sigma.permute(0, 2, 3, 1).contiguous().view(-1, num_feat)
        label = label.contiguous().view(-1)

        mask = (label != 255)
        label = label[mask]
        sigma = sigma[mask]
        mu = mu[mask]
        propotion = torch.sum(mask) / mask.size()[0]
        


        mu_hard_list = []
        sigma_hard_list = []
        num_list = []
        class_list = []


        for i in range(class_num):  # 19
            # valid_pixel = valid_pixel_all[:, i]  # 就是类别Mask
            class_mask = (label == i)

            if class_mask.sum() == 0:
                num_list.append(0)
                mu_hard_list.append(0)
                sigma_hard_list.append(0)
                continue
           

            num_list.append(int(class_mask.sum().item()))  
            class_list.append(i)

        # Compute Probabilistic Representation Contrastive Loss
        if (len(num_list) <= 1):  # in some rare cases, a small mini-batch only contain 1 or no semantic class
            return torch.tensor(0.0)
        else:
            pppc_loss = torch.tensor(0.0)


            for i in range(class_num):
                if num_list[i] == 0:
                    continue
               
                # else:
                #     continue
                class_mask = (label == i)
                anchor_mu = mu[class_mask]
                anchor_sigma = sigma[class_mask]
                with torch.no_grad():
                    
                    all_mu = proto_mu.repeat(num_list[i], 1, 1)
                    all_sigma = proto_sigma.repeat(num_list[i], 1, 1)

                logits = ELK(anchor_mu.unsqueeze(1), all_mu, anchor_sigma.unsqueeze(1), all_sigma)
                pppc_loss = pppc_loss + F.cross_entropy(logits / self.temp,
                                                        (torch.ones(num_list[i])*i).long().cuda())

            pppc_loss = (pppc_loss / len(class_list))


            vib_loss = \
                self.kl_divergence(mu, sigma) * 1e-6
            loss_total = pppc_loss + vib_loss

            log_vars[f"{self.domain}_prcl"] = pppc_loss.detach().cpu().numpy()
            log_vars[f"{self.domain}_vib"] = vib_loss.detach().cpu().numpy()
            log_vars[f"{self.domain}_total_without_scale"] = loss_total.detach().cpu().numpy()


            return loss_total * self.scale_factor


#### Utils ####
def negative_index_sampler(samp_num, seg_num_list):
  
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            low = sum(seg_num_list[: j])
            high = sum(seg_num_list[: j + 1])
            size = int(samp_num[i, j])
            negative_index += np.random.randint(low, high, size).tolist()

    return negative_index


#### MLS ####
def ELK(mu_0, mu_1, sigma_0, sigma_1):
    '''
    Compute the MLS
    param: mu_0, mu_1 [256, 513, 256]  [256, 1, 256]  
           sigma_0, sigma_1 [256, 513, 256] [256, 1, 256]
    '''
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)
    up = (mu_0 - mu_1) ** 2
    down = sigma_0.exp() + sigma_1.exp()
    mls = -0.5 * (up / down + torch.log(down)).mean(-1)

    return mls

def BK_score(mu_0, mu_1, sigma_0, sigma_1):
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)
    up = (mu_0 - mu_1) ** 2
    down = sigma_0.exp() + sigma_1.exp()
    sigma_qr0 = torch.sqrt(sigma_0.exp() + 1e-6)
    sigma_qr1 = torch.sqrt(sigma_1.exp() + 1e-6)
    log_up = sigma_qr0 * sigma_qr1
    bk = (-0.25 * (up / down) + 0.5 * torch.log(log_up / down)).mean(-1)
    return bk
def w_distance(mu_0, mu_1, sigma_0, sigma_1):


    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)


    # w- distance
    left = (mu_0 - mu_1) ** 2
    right = torch.sqrt(sigma_0.exp() + 1e-6) -  sigma_1.exp()
    w_dis = left + right

    return -w_dis.mean(-1)

def KL_div(mu_0, mu_1, sigma_0, sigma_1):
    mu_0 = F.normalize(mu_0, dim=-1)
    mu_1 = F.normalize(mu_1, dim=-1)
    sigma_0 = sigma_0.exp()
    sigma_1 = sigma_1.exp()
    eq1 = (sigma_1 / sigma_0).log()
    eq2 = sigma_0 / sigma_1
    eq3 = (mu_0 - mu_1) ** 2 / sigma_1
    kl = eq1 + eq2 + eq3
    return -1/2 * kl.mean(-1)

def js_div(mu_0, mu_1, sigma_0, sigma_1):
    return 1/2 * (KL_div(mu_0, mu_1, sigma_0, sigma_1) + KL_div(mu_1, mu_0, sigma_1, sigma_0))