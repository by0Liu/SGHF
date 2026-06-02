import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet

from stage2_sghf.utils.tool import compute_AUCs
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from stage2_sghf.models.ham import HAM_CXR_14
from custom_dataset import NIHDataset
from torch.optim.lr_scheduler import StepLR


Class_Names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
           'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

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

    transform_test = transforms.Compose([transforms.Resize(224),
                                         # transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join("/media/user/0291ebd7-713f-4d1c-9a78-ac6224af6b10/home/cvnlp/archive/chext dataset"))
    image_path = os.path.join(data_root, 'images')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_list_path = os.path.join(data_root,'csv', 'train_list.txt')
    assert os.path.exists(train_list_path), "{} path does not exist.".format(train_list_path)
    valid_list_path = os.path.join(data_root,'csv', 'valid_list.txt')
    assert os.path.exists(valid_list_path), "{} path does not exist.".format(valid_list_path)
    test_list_path = os.path.join(data_root, 'csv', 'test_list.txt')
    assert os.path.exists(test_list_path), "{} path does not exist.".format(test_list_path)

    train_data = NIHDataset(data_path=image_path, image_list_path=train_list_path, transform=transform_train)
    valid_data = NIHDataset(data_path=image_path, image_list_path=valid_list_path, transform=transform_valid)
    test_data = NIHDataset(data_path=image_path, image_list_path=test_list_path, transform=transform_test)

    Batch_Size = 16

    train_loader = DataLoader(train_data, batch_size=Batch_Size, shuffle=True, num_workers=8, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=Batch_Size, shuffle=False, num_workers=8, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=Batch_Size, shuffle=False, num_workers=8, drop_last=True)

    train_num = len(train_data)
    valid_num = len(valid_data)
    test_num = len(test_data)

    print("using {} images for training, {} images for validation, {} images for testing.".format(train_num, valid_num,
                                                                                                  test_num))

    eye_prenet = eyenet(num_classes=14,backbone='resnet101',temp=1.0,fpn_out=512,head_mid=256,use_aspp=True)
    eye_prenet_weight_path = "gaze_fpnv2_best_total.pth"
    assert os.path.exists(eye_prenet_weight_path), "file {} does not exist.".format(eye_prenet_weight_path)



    eye_prenet.load_state_dict(torch.load(eye_prenet_weight_path, map_location=device))
    print('eyenet.pth is load')
    eye_prenet.to(device)

    HAM = HAM_CXR_14()

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

    loss_function = AsymmetricLoss()


    params = list(net.parameters()) + list(HAM.parameters()) + list(vit_b.parameters())

    optimizer = optim.Adam(params, lr=0.00005)   # 0.00005
    # step_schedule = optim.lr_scheduler.StepLR(step_size=10,gamma=0.9,optimizer=optimizer)
    scheduler = StepLR(optimizer,gamma = 0.9,last_epoch = -1,step_size = 10)

    epochs = 30
    best_auc = 0.0
    best_val_auc = 0.0
    save_path = './res_net.pth'
    save_path1 = './ham.pth'
    save_path2 = './vit_b.pth'
    train_steps = len(train_loader)

    train_loss = []
    val_auc = []
    test_auc = []

    for epoch in range(epochs):
        # train
        eye_prenet.eval()
        net.train()

        vit_b.train()

        HAM.train()
        running_loss = 0.0

        tgt = torch.FloatTensor()
        tgt = tgt.to(device)
        gpred = torch.FloatTensor()
        gpred = gpred.to(device)


        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            tgt = torch.cat((tgt, labels), 0)

            with torch.no_grad():
                eye_cam, M, logits1 = eye_prenet(images)


            optimizer.zero_grad()

            cat_feature = HAM(images.to(device), net, vit_b, eye_cam)
            gpred = torch.cat((gpred, cat_feature.data), 0)
            loss = loss_function(cat_feature, labels)

            train_loss.append(loss.item())

            with open("train_loss.txt", 'w') as train_los:
                train_los.write(str(train_loss))

            loss.backward()
            optimizer.step()
            # step_schedule.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(
                epoch + 1, epochs, loss)

        aucc_g = compute_AUCs(tgt, gpred)
        aucc_avg_g = np.array(aucc_g).mean()
        print('[epoch %d] train_global_mean_auc: %.3f' %
              (epoch + 1, aucc_avg_g))

        gt = torch.FloatTensor()
        gt = gt.to(device)
        pred = torch.FloatTensor()
        pred = pred.to(device)

        scheduler.step()
        print('Finished Training')

        # valid
        eye_prenet.eval()
        net.eval()
        vit_b.eval()
        HAM.eval()
        # acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(valid_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)
                gt = torch.cat((gt, val_labels), 0)

                eye_cam, M, logits1 = eye_prenet(val_images)

                cat_feature = HAM(val_images, net, vit_b, eye_cam)
                pred = torch.cat((pred, cat_feature.data), 0)

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()

        val_auc.append(AUROC_avg.item())
        with open("./valid_mean_auc.txt", 'w') as train_ac:
            train_ac.write(str(val_auc))

        print('Finished valid')

        # test

        gt_test = torch.FloatTensor()
        gt_test = gt_test.to(device)
        pred_test = torch.FloatTensor()
        pred_test = pred_test.to(device)

        eye_prenet.eval()
        net.eval()

        vit_b.eval()

        HAM.eval()

        # acc = 0.0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for test_data in test_bar:
                test_images, test_labels = test_data
                test_labels = test_labels.to(device)
                test_images = test_images.to(device)
                gt_test = torch.cat((gt_test, test_labels), 0)

                eye_cam, M, logits1 = eye_prenet(test_images)

                cat_feature = HAM(test_images, net, vit_b, eye_cam)
                pred_test = torch.cat((pred_test, cat_feature.data), 0)

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1, epochs)

        AUROCs_test = compute_AUCs(gt_test, pred_test)
        AUROC_avg_test = np.array(AUROCs_test).mean()

        test_auc.append(AUROC_avg_test.item())

        with open("./test_mean_auc.txt", 'w') as test_ac:
            test_ac.write(str(test_auc))

        print('[epoch %d] train_loss: %.3f valid_mean_auc: %.3f test_mean_auc: %.3f' %
              (epoch + 1, running_loss / train_steps, AUROC_avg, AUROC_avg_test))

        print('[epoch %d] valid_mean_auc: %.3f' %
              (epoch + 1, AUROC_avg))
        for i in range(14):
            print('The val_AUROC of {} is {}'.format(Class_Names[i], AUROCs[i]))

        print('[epoch %d] test_mean_auc: %.3f' %
              (epoch + 1, AUROC_avg_test))
        for i in range(14):
            print('The test_AUROC of {} is {}'.format(Class_Names[i], AUROCs_test[i]))

        if AUROC_avg_test > best_auc:
            best_auc = AUROC_avg_test
            torch.save(net.state_dict(), save_path)
            torch.save(HAM.state_dict(), save_path1)
            torch.save(vit_b.state_dict(), save_path2)
            print('保存第' + str(epoch) + '次的权重')


    print('ALL OVER')

if __name__ == '__main__':
    main()

