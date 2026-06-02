import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
import cv2
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.ham import HAM_CXR
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from custom_dataset import MedicalImageDataset,read_split_data
from torch.optim.lr_scheduler import StepLR

Class_Names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']



class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=2, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


def main():


    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed_all(3407)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    torch.backends.cudnn.benchmark = True

    transform_train = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_valid = transforms.Compose([transforms.Resize(224),
                                          # transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    data_path = "./dataset/COVID-19_Radiography_Dataset"

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(data_path)

    train_data = MedicalImageDataset(data_path=train_images_path, data_label=train_images_label, transform=transform_train)
    valid_data = MedicalImageDataset(data_path=val_images_path, data_label=val_images_label, transform=transform_valid)

    Batch_Size = 2
    Batch_Size2 = 2

    train_loader = DataLoader(train_data, batch_size=Batch_Size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=Batch_Size2, shuffle=False, num_workers=8, drop_last=True)

    train_num = len(train_data)
    valid_num = len(valid_data)

    print("using {} images for training, {} images for validation.".format(train_num, valid_num))

    eye_prenet = eyenet(num_classes=14,backbone='resnet101',temp=1.0,fpn_out=512,head_mid=256,use_aspp=True)
    eye_prenet_weight_path = "gaze_fpnv2_best_total.pth"
    assert os.path.exists(eye_prenet_weight_path), "file {} does not exist.".format(eye_prenet_weight_path)


    eye_prenet.load_state_dict(torch.load(eye_prenet_weight_path, map_location=device))
    print('eyenet.pth is load')
    eye_prenet.to(device)

    HAM = HAM_CXR()

    HAM.apply(initialize_weights)
    HAM.to(device)

    vit_b = vit()
    vit_b.to(device)


    net = resnet50(stride_list=[1,2,1,1],use_maxpooling=True,dilations=(1,1,2,4),norm_layer=None)

    net_dict = net.state_dict()
    predict_model = torch.load('resnet50-19c8e357.pth')
    state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)

    net.to(device)


    loss_function = ASLSingleLabel()
    params = list(net.parameters()) + list(HAM.parameters()) + list(vit_b.parameters())
    optimizer = optim.Adam(params, lr=0.00005)
    scheduler = StepLR(optimizer,gamma = 0.9,last_epoch = -1,step_size = 10)

    epochs = 1
    save_path = './res_net.pth'
    save_path1 = './ham.pth'
    save_path2 = './vit_b.pth'
    best_acc1 = 0.



    for epoch in range(epochs):

        Label_train = []
        Train_acc1 = []
        Label_val = []
        Val_acc1 = []

        # train
        eye_prenet.eval()
        net.train()
        vit_b.train()
        HAM.train()

        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            labels_train = labels.tolist()
            for i in range(Batch_Size):
                Label_train.append(labels_train[i])

            with torch.no_grad():
                eye_cam, M, logits1 = eye_prenet(images)

            optimizer.zero_grad()
            cat_feature = HAM(images.to(device), net, vit_b, eye_cam)

            pred_classes1 = torch.max(cat_feature, dim=1)[1]
            pred_classes2 = pred_classes1.tolist()
            for i in range(Batch_Size):
                Train_acc1.append(pred_classes2[i])
            loss_f = loss_function(cat_feature, labels.to(device))

            loss = loss_f

            loss.backward()

            train_bar.desc = "[model_train epoch {}] ,lr: {:.5f}".format(
                epoch + 1, optimizer.param_groups[0]["lr"])

            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)

            optimizer.step()

        scheduler.step()
        print('Finished Training')

        # valid
        eye_prenet.eval()
        net.eval()
        vit_b.eval()
        HAM.eval()

        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout)
            for step, val_data in enumerate(val_bar):
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                labels_val = val_labels.tolist()
                for i in range(Batch_Size2):
                    Label_val.append(labels_val[i])

                eye_cam, M, logits1 = eye_prenet(val_images)

                cat_feature = HAM(val_images.to(device), net, vit_b, eye_cam)
                pred_classes1 = torch.max(cat_feature, dim=1)[1]
                pred_classes2 = pred_classes1.tolist()
                for i in range(Batch_Size2):
                    Val_acc1.append(pred_classes2[i])

                val_bar.desc = "[model_valid epoch {}]".format(
                    epoch + 1)

        accuracy_train = accuracy_score(Label_train, Train_acc1)
        accuracy_val = accuracy_score(Label_val, Val_acc1)
        precision_train = precision_score(Label_train, Train_acc1, labels=[0, 1, 2, 3], average='macro')
        precision_val = precision_score(Label_val, Val_acc1, labels=[0, 1, 2, 3], average='macro')
        recall_train = recall_score(Label_train, Train_acc1, labels=[0, 1, 2, 3], average='macro')
        recall_val = recall_score(Label_val, Val_acc1, labels=[0, 1, 2, 3], average='macro')
        f1_train = f1_score(Label_train, Train_acc1, labels=[0, 1, 2, 3], average='macro')
        f1_val = f1_score(Label_val, Val_acc1, labels=[0, 1, 2, 3], average='macro')

        if best_acc1 < accuracy_val:
            torch.save(net.state_dict(), save_path)
            torch.save(HAM.state_dict(), save_path1)
            torch.save(vit_b.state_dict(), save_path2)
            best_acc1 = accuracy_val

        print(
            '[epoch %d] accuracy_train: %.6f \n precision_train: %.6f \n recall_train: %.6f \n f1_train: %.6f \n accuracy_val: %.6f \n precision_val: %.6f \n recall_val: %.6f \n f1_val: %.6f' %
            (
            epoch + 1, accuracy_train, precision_train, recall_train, f1_train, accuracy_val, precision_val, recall_val,
            f1_val))

    print("The Best Result :")
    print("best acc1:{}".format(best_acc1))


if __name__ == '__main__':
    main()





