# -*- coding: utf-8 -*-
import os, sys, math, cv2, numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101


# ================== AUC ==================
def compute_AUCs(targets_t, preds_t):
    try:
        from sklearn.metrics import roc_auc_score
        targets = targets_t.detach().cpu().numpy()
        preds   = preds_t.detach().cpu().numpy()
        C = targets.shape[1]
        aucs = []
        for c in range(C):
            y_true = targets[:, c]
            y_score = preds[:, c]
            if (y_true.max() == y_true.min()):
                aucs.append(float("nan"))
            else:
                aucs.append(roc_auc_score(y_true, y_score))
        return aucs
    except Exception as e:
        print("[WARN] AUC failed:", e)
        return [float("nan")] * preds_t.shape[1]

# ================== FPN & ASPP ==================
class FPN(nn.Module):
    def __init__(self, C2=256, C3=512, C4=1024, C5=2048, out=256):
        super().__init__()
        self.l2 = nn.Conv2d(C2, out, 1)
        self.l3 = nn.Conv2d(C3, out, 1)
        self.l4 = nn.Conv2d(C4, out, 1)
        self.l5 = nn.Conv2d(C5, out, 1)
        self.s2 = nn.Conv2d(out, out, 3, padding=1)
        self.s3 = nn.Conv2d(out, out, 3, padding=1)
        self.s4 = nn.Conv2d(out, out, 3, padding=1)
        self.s5 = nn.Conv2d(out, out, 3, padding=1)

    def forward(self, c2, c3, c4, c5):
        p5 = self.l5(c5)
        p4 = self.l4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='nearest')
        p3 = self.l3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='nearest')
        p2 = self.l2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='nearest')
        return self.s2(p2), self.s3(p3), self.s4(p4), self.s5(p5)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, rates=(1, 6, 12, 18)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        ])
        for r in rates[1:]:
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(True)
            ))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch*(len(rates)+1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(True)
        )

    def forward(self, x):
        H, W = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        gp = self.image_pool(x)
        gp = F.interpolate(gp, size=(H, W), mode='bilinear', align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        return self.project(x)

# ================== Model ==================
class GazeFPNPlus(nn.Module):
    """
      Q_g: [B,1,H,W]
      M  : [B,C,H,W]
      y_logits: [B,C]
    """
    def __init__(self, num_classes, backbone='resnet50', temp=1.0,
                 fpn_out=256, head_mid=128, use_aspp=False):
        super().__init__()
        self.temp = temp
        if backbone == 'resnet50':
            net = resnet50(weights='IMAGENET1K_V2')
        else:
            net = resnet101(weights='IMAGENET1K_V2')

        # stem + C2..C5
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = net.layer1, net.layer2, net.layer3, net.layer4

        self.fpn = FPN(256, 512, 1024, 2048, out=fpn_out)

        self.h2 = nn.Sequential(nn.Conv2d(fpn_out, head_mid, 3, padding=1), nn.ReLU(True))
        self.h3 = nn.Sequential(nn.Conv2d(fpn_out, head_mid, 3, padding=1), nn.ReLU(True))
        self.h4 = nn.Sequential(nn.Conv2d(fpn_out, head_mid, 3, padding=1), nn.ReLU(True))
        self.h5 = nn.Sequential(nn.Conv2d(fpn_out, head_mid, 3, padding=1), nn.ReLU(True))

        self.use_aspp = use_aspp
        if use_aspp:
            self.aspp = ASPP(head_mid, head_mid)

        self.gaze_head   = nn.Conv2d(head_mid, 1, 1)
        self.lesion_head = nn.Conv2d(head_mid, num_classes, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = self.stem(x)
        c2 = self.layer1(x); c3 = self.layer2(c2); c4 = self.layer3(c3); c5 = self.layer4(c4)

        p2, p3, p4, p5 = self.fpn(c2, c3, c4, c5)
        base = self.h2(p2) \
             + F.interpolate(self.h3(p3), size=p2.shape[-2:], mode='bilinear', align_corners=False) \
             + F.interpolate(self.h4(p4), size=p2.shape[-2:], mode='bilinear', align_corners=False) \
             + F.interpolate(self.h5(p5), size=p2.shape[-2:], mode='bilinear', align_corners=False)
        if self.use_aspp:
            base = self.aspp(base)

        H_g = self.gaze_head(base)                           # [B,1,H/4,W/4]
        H_g = F.interpolate(H_g, size=(H, W), mode='bicubic', align_corners=False)
        Q_g = torch.softmax((H_g / self.temp).flatten(2), dim=-1).view_as(H_g)

        M = self.lesion_head(base)                           # [B,C,H/4,W/4]
        M = F.interpolate(M, size=(H, W), mode='bilinear', align_corners=False)

        y_logits = mil_logsumexp_pool_logits(M, tau=10.0, normalize=True)
        return Q_g, M, y_logits

# ================== MIL ==================
def mil_logsumexp_pool_logits(M, tau=10.0, normalize=True):
    B, C, H, W = M.shape
    m = M.view(B, C, -1)
    lse = torch.logsumexp(tau * m, dim=-1) / tau
    if normalize:
        lse = lse - math.log(H * W)
    return lse

# ================== Loss (KL + CC) ==================
def kl_divergence(P, Q, eps=1e-8):
    return (P * (torch.log(P + eps) - torch.log(Q + eps))).sum(dim=(1,2,3)).mean()

def cc_value(Q, P, eps=1e-8):
    Qc = Q - Q.mean(dim=(2,3), keepdim=True)
    Pc = P - P.mean(dim=(2,3), keepdim=True)
    num = (Qc * Pc).sum(dim=(2,3))
    den = torch.sqrt((Qc**2).sum(dim=(2,3)) * (Pc**2).sum(dim=(2,3)) + eps)
    cc = num / (den + eps)
    return cc.mean()

def gaze_loss(Q, P, w_kl=1.0, w_cc=1.0):
    L_kl = kl_divergence(P, Q)
    CC   = cc_value(Q, P)
    return w_kl * L_kl + w_cc * (1.0 - CC), L_kl.detach(), CC.detach()

# ================== Utils ==================
def load_eyegaze_map(data_root, file_rel_path, out_hw):
    path = os.path.join(data_root, "eye_cam", file_rel_path, "eye_cam.npy")
    eye_cam = np.load(path, allow_pickle=True).item()['np_image']
    H, W = out_hw
    em = cv2.resize(eye_cam, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
    em = em - em.min()
    if em.max() > 0:
        em = em / em.max()
    s = em.sum()
    if s <= 1e-6:
        em[:] = 1.0 / (H * W)
    else:
        em = em / s
    return torch.from_numpy(em)[None, ...]  # [1,H,W]

# ================== Train / Valid ==================
def train_one_epoch(net, train_loader, optimizer, device, data_root):
    net.train()
    bce = nn.BCEWithLogitsLoss()
    running = {'gaze':0.0, 'cls':0.0, 'total':0.0, 'kl':0.0, 'cc':0.0}
    tgt_all, pred_all = [], []
    pbar = tqdm(train_loader, file=sys.stdout)

    steps = 0
    for images, labels, filepath in pbar:
        images = images.to(device)
        labels = labels.to(device).float()
        B, _, H, W = images.shape

        Q_g, M, y_logits = net(images)

        P_batch = torch.stack(
            [load_eyegaze_map(data_root, filepath[i], (H, W)) for i in range(B)],
            dim=0
        ).to(device)

        L_gaze, L_kl, CC = gaze_loss(Q_g, P_batch, w_kl=1.0, w_cc=1.0)
        L_cls = bce(y_logits, labels)
        loss = L_gaze + L_cls

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps += 1
        running['gaze'] += L_gaze.item()
        running['cls']  += L_cls.item()
        running['total']+= loss.item()
        running['kl']   += L_kl.item()
        running['cc']   += CC.item()

        tgt_all.append(labels.detach().float().cpu())
        pred_all.append(torch.sigmoid(y_logits.detach()).cpu())

        avg_kl = running['kl']/steps
        avg_cc = running['cc']/steps
        pbar.desc = f"train: gaze {running['gaze']/steps:.3f} | cls {running['cls']/steps:.3f} | KL {avg_kl:.3f} | CC {avg_cc:.3f}"

    tgt_all = torch.cat(tgt_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    aucs = compute_AUCs(tgt_all, pred_all)
    return running, aucs, steps

@torch.no_grad()
def valid_one_epoch(net, valid_loader, device, data_root):
    net.eval()
    bce = nn.BCEWithLogitsLoss()
    running = {'gaze':0.0, 'cls':0.0, 'total':0.0, 'kl':0.0, 'cc':0.0}
    tgt_all, pred_all = [], []
    pbar = tqdm(valid_loader, file=sys.stdout)

    steps = 0
    for images, labels, filepath in pbar:
        images = images.to(device)
        labels = labels.to(device).float()
        B, _, H, W = images.shape

        Q_g, M, y_logits = net(images)
        P_batch = torch.stack(
            [load_eyegaze_map(data_root, filepath[i], (H, W)) for i in range(B)],
            dim=0
        ).to(device)

        L_gaze, L_kl, CC = gaze_loss(Q_g, P_batch, w_kl=1.0, w_cc=1.0)
        L_cls = bce(y_logits, labels)
        loss = L_gaze + L_cls

        steps += 1
        running['gaze'] += L_gaze.item()
        running['cls']  += L_cls.item()
        running['total']+= loss.item()
        running['kl']   += L_kl.item()
        running['cc']   += CC.item()

        tgt_all.append(labels.detach().float().cpu())
        pred_all.append(torch.sigmoid(y_logits.detach()).cpu())

        avg_kl = running['kl']/steps
        avg_cc = running['cc']/steps
        pbar.desc = f"valid: gaze {running['gaze']/steps:.3f} | cls {running['cls']/steps:.3f} | KL {avg_kl:.3f} | CC {avg_cc:.3f}"

    tgt_all = torch.cat(tgt_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    aucs = compute_AUCs(tgt_all, pred_all)
    return running, aucs, steps

