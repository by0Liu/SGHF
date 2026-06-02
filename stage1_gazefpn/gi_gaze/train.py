import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.backends import cudnn
import torch
import os
from dataset import GazeDataset, image_batch_wrapper, rand_flip_rotate_augmentation
from torch.utils.data import DataLoader
from tqdm import tqdm
from stage1_gazefpn.models.gazefpn import GazeFPNPlus, gaze_loss

def sanitize_gaze_tensor(gaze, expect_size=None):

    g = gaze.float()
    if g.dim()==3: g = g.unsqueeze(1)          # [B,1,H,W]
    if g.size(1)>1:

        g = g.mean(dim=1, keepdim=True)
    g = g - g.amin(dim=(2,3), keepdim=True)
    denom = (g.amax(dim=(2,3), keepdim=True) - g.amin(dim=(2,3), keepdim=True)).clamp_min(1e-8)
    g = g / denom
    if expect_size is not None:
        g = F.interpolate(g, size=expect_size, mode='bilinear', align_corners=False)
        g = g - g.amin(dim=(2,3), keepdim=True)
        g = g / (g.amax(dim=(2,3), keepdim=True).clamp_min(1e-8))
    return g

def to_probability_map(g, eps=1e-8, on_zero="uniform"):

    S = g.sum(dim=(2,3), keepdim=True)
    zero_mask = (S <= eps)
    B,_,H,W = g.shape
    if on_zero == "skip" and zero_mask.any():
        return None
    g_prob = torch.where(zero_mask, torch.full_like(g, 1.0/(H*W)), g / (S + eps))
    return g_prob


@torch.no_grad()
def cls_eval(model, dataloader, num_classes=14, device=None):

    model.eval()
    bce = nn.BCEWithLogitsLoss()

    total_samples = 0
    correct_count = 0
    multilabel_match_sum = 0

    bce_loss_total = 0.0
    gaze_loss_total = 0.0
    kl_total = 0.0
    cc_total = 0.0
    n_batches = 0

    for batch in dataloader:
        if device is not None:
            img = batch.img.to(device)
            gaze = batch.gaze.to(device)
            label = batch.label.to(device)
        else:
            img, gaze, label = batch.img, batch.gaze, batch.label

        B = img.size(0)
        total_samples += B
        n_batches += 1


        Q_g, M, y_logits = model(img)  # Q_g:[B,1,H,W]; y_logits:[B,C]
        B, _, H, W = img.shape
        P = sanitize_gaze_tensor(gaze, expect_size=(H, W))
        P = to_probability_map(P, on_zero="uniform")

        L_gaze, L_kl, CC = gaze_loss(Q_g, P, w_kl=1.0, w_cc=1.0)

        if label.dim() == 1:
            label_oh = F.one_hot(label.long(), num_classes=num_classes).float()
        else:
            label_oh = label.float()
        L_cls = bce(y_logits, label_oh)

        bce_loss_total  += L_cls.item()
        gaze_loss_total += L_gaze.item()
        kl_total        += L_kl.item()
        cc_total        += CC.item()


        probs = torch.sigmoid(y_logits)
        if label.dim() == 1:
            pred_cls = probs.argmax(dim=1)
            correct_count += (pred_cls == label.long()).sum().item()
        else:
            pred_multi = (probs > 0.5).float()
            multilabel_match_sum += (pred_multi == label_oh).float().mean().item()

    if total_samples == 0 or n_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if label.dim() == 1:
        acc = correct_count / total_samples
    else:
        acc = multilabel_match_sum / n_batches

    bce_loss_avg  = bce_loss_total / n_batches
    gaze_loss_avg = gaze_loss_total / n_batches
    kl_avg        = kl_total / n_batches
    cc_avg        = cc_total / n_batches
    return acc, bce_loss_avg, gaze_loss_avg, kl_avg, cc_avg


def train(seed, train_dataset, val_dataset, net, model_name, use_gaze, deterministic=True,
          save_root='./checkpoints/'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()
    # reproduce
    if not deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    def worker_init_fn(worker_id, seed=seed):
        random.seed(seed + worker_id)

    train_batch_size = 8
    val_batch_size = 16
    train_loader = DataLoader(train_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              collate_fn=image_batch_wrapper,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            collate_fn=image_batch_wrapper,
                            pin_memory=True)

    best_metric = 0.7
    best_gaze_score = float('inf')

    lr = 5e-4

    net.to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    save_path = os.path.join(save_root, model_name)
    os.makedirs(save_path, exist_ok=True)

    max_epoch = 150
    gaze_loss_factor = 1
    if not use_gaze:
        gaze_loss_factor = 0
    ce_loss_factor = 1
    hyperparams = {'model_name': model_name, 'train_batch_size': train_batch_size,
                   'max_epoch': max_epoch, 'lr': lr, 'gaze_loss_factor': gaze_loss_factor,
                   'ce_loss_factor': ce_loss_factor}
    print(hyperparams)
    logger = open(os.path.join(save_path, 'logs.txt'), "w")
    logger.write('\n' + str(hyperparams))

    iterator = tqdm(range(max_epoch), ncols=70)
    bce = nn.BCEWithLogitsLoss()

    for epoch_num in iterator:
        net.train()
        train_loss = 0.0

        for i_batch, batch in enumerate(train_loader):
            if cuda:
                img   = batch.img.cuda()
                gaze  = batch.gaze.cuda()
                label = batch.label.cuda()
            else:
                img, gaze, label = batch.img, batch.gaze, batch.label

            B, _, H, W = img.shape
            Q_g, M, y_logits = net(img)


            P = sanitize_gaze_tensor(gaze, expect_size=(H, W))
            P = to_probability_map(P, on_zero="uniform")

            if label.dim() == 1:
                label_oh = F.one_hot(label.long(), num_classes=M.shape[1]).float()
            else:
                label_oh = label.float()

            L_gaze, _, _ = gaze_loss(Q_g, P, w_kl=1.0, w_cc=1.0)
            L_cls = bce(y_logits, label_oh)
            loss = L_gaze + L_cls

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(1, len(train_loader))
        scheduler.step()

        val_acc, val_bce_loss, val_gaze_loss, val_kl, val_cc = cls_eval(
            net, val_loader, num_classes= M.shape[1], device=device
        )
        gaze_score = val_kl + (1.0 - val_cc)

        lr_ = optimizer.param_groups[0]['lr']

        # 保存：按 acc
        if val_acc >= best_metric:
            best_metric = val_acc
            save_name = os.path.join(save_path, model_name + '_best_model.pth')
            torch.save(net.state_dict(), save_name)
            info = ('BestACC: epoch %d, lr: %.6f, val_acc: %.4f, val_bce: %.4f, val_gaze: %.4f, '
                    'val_kl: %.4f, val_cc: %.4f, gaze_score: %.4f, train_loss: %.4f' %
                    (epoch_num, lr_, val_acc, val_bce_loss, val_gaze_loss, val_kl, val_cc, gaze_score, train_loss))
            logger.write('\n' + info)
        else:
            info = ('epoch %d, lr: %.6f, val_acc: %.4f, val_bce: %.4f, val_gaze: %.4f, '
                    'val_kl: %.4f, val_cc: %.4f, gaze_score: %.4f, train_loss: %.4f' %
                    (epoch_num, lr_, val_acc, val_bce_loss, val_gaze_loss, val_kl, val_cc, gaze_score, train_loss))
            logger.write('\n' + info)
        print(info)

        if gaze_score < best_gaze_score:
            best_gaze_score = gaze_score
            save_name_gaze = os.path.join(save_path, model_name + '_best_gaze.pth')
            torch.save(net.state_dict(), save_name_gaze)
            print(f'  -> Save best (GazeScore) to {save_name_gaze}')

    print('Finished Training')


if __name__ == "__main__":
    torch.cuda.set_device(0)
    deterministic = True
    seed = 0

    print('loading datasets')
    dataset = GazeDataset(
        folder_path='data/train/',
        transform=rand_flip_rotate_augmentation
    )
    dataset_val = GazeDataset(
        folder_path='data/val/'
    )
    print('datasets loaded')
    print('Number of images in training dataset:', len(dataset))
    print('Number of images in validation dataset:', len(dataset_val))

    net = GazeFPNPlus(num_classes=3, backbone='resnet101', temp=1.0, fpn_out=512, head_mid=256, use_aspp=True)

    model_name = 'Colorectal Polyp_' + 'resnet101'
    train(seed, dataset, dataset_val, net, model_name, deterministic=deterministic,
          save_root='./checkpoints/')


