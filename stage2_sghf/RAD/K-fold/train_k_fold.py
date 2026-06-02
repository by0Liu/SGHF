import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import torch
import numpy as np
from split_data import read_all_data
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.ham import HAM_CXR
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import StepLR
from stage2_sghf.RAD.train import initialize_weights,ASLSingleLabel
from sklearn.model_selection import StratifiedKFold
from stage2_sghf.RAD.custom_dataset import MedicalImageDataset,read_split_data


def build_one_fold_loaders(all_paths, all_labels, train_idx, val_idx,
                           transform_train, transform_valid,
                           batch_size_train=16, batch_size_val=16, num_workers=4):
    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_paths   = [all_paths[i] for i in val_idx]
    val_labels  = [all_labels[i] for i in val_idx]

    train_data = MedicalImageDataset(data_path=train_paths, data_label=train_labels, transform=transform_train)
    valid_data = MedicalImageDataset(data_path=val_paths,   data_label=val_labels,   transform=transform_valid)

    train_loader = DataLoader(train_data, batch_size=batch_size_train, shuffle=True,
                              num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_val, shuffle=False,
                              num_workers=num_workers, drop_last=False)

    return train_loader, valid_loader, len(train_data), len(valid_data)

def init_models_and_optim(device):
    # -------------- eyenet --------------
    eye_prenet = eyenet(num_classes=3, backbone='resnet101', temp=1.0,
                        fpn_out=512, head_mid=256, use_aspp=True)

    eye_prenet_weight_path = "../gaze_fpnv2_best_total.pth"
    assert os.path.exists(eye_prenet_weight_path), f"file {eye_prenet_weight_path} does not exist."
    eye_prenet.load_state_dict(torch.load(eye_prenet_weight_path, map_location=device))
    eye_prenet.to(device)
    eye_prenet.eval()

    # -------------- HAM / ViT / ResNet --------------
    HAM = HAM_CXR()
    HAM.apply(initialize_weights)
    HAM.to(device)

    vit_b = vit()
    vit_b.to(device)

    net = resnet50(stride_list=[1,2,1,1], use_maxpooling=True, dilations=(1,1,2,4), norm_layer=None)


    net_dict = net.state_dict()
    predict_model = torch.load('../resnet50-19c8e357.pth', map_location="cpu")
    state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
    net_dict.update(state_dict)
    net.load_state_dict(net_dict)
    net.to(device)

    # -------------- loss / optim / scheduler --------------
    loss_function = ASLSingleLabel()
    params = list(net.parameters()) + list(HAM.parameters()) + list(vit_b.parameters())
    optimizer = optim.Adam(params, lr=0.00005)
    scheduler = StepLR(optimizer, gamma=0.9, last_epoch=-1, step_size=10)

    return eye_prenet, net, vit_b, HAM, loss_function, optimizer, scheduler


def compute_metrics(y_true, y_pred, num_classes=4):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=list(range(num_classes)), average='macro', zero_division=0)
    return acc, precision, recall, f1


def main_kfold():
    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed_all(3407)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    torch.backends.cudnn.benchmark = True

    # ----------------- transforms -----------------
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_valid = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    data_path = "./dataset/COVID-19_Radiography_Dataset"

    all_paths, all_labels, class_indices = read_all_data(data_path, expected_num_classes=4)
    all_labels_np = np.array(all_labels)

    K = 4
    epochs = 50
    Batch_Size = 16
    Batch_Size2 = 16
    num_workers = 4
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=3407)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(all_labels_np)), all_labels_np), start=1):
        print(f"\n==================== Fold {fold}/{K} ====================")

        train_loader, valid_loader, train_num, valid_num = build_one_fold_loaders(
            all_paths, all_labels, train_idx, val_idx,
            transform_train, transform_valid,
            batch_size_train=Batch_Size, batch_size_val=Batch_Size2,
            num_workers=num_workers
        )
        print(f"using {train_num} images for training, {valid_num} images for validation.")

        eye_prenet, net, vit_b, HAM, loss_function, optimizer, scheduler = init_models_and_optim(device)

        save_path_net = f'./ckpt_fold{fold}_res_net.pth'
        save_path_ham = f'./ckpt_fold{fold}_ham.pth'
        save_path_vit = f'./ckpt_fold{fold}_vit_b.pth'

        best_acc = 0.0

        for epoch in range(epochs):
            # ----------------- Train -----------------
            Label_train, Train_pred = [], []

            eye_prenet.eval()
            net.train(); vit_b.train(); HAM.train()

            train_bar = tqdm(train_loader, file=sys.stdout)
            for step, data in enumerate(train_bar):
                images, labels = data
                images = images.to(device)
                labels = labels.to(device)

                Label_train.extend(labels.detach().cpu().tolist())

                with torch.no_grad():
                    eye_cam, M, logits1 = eye_prenet(images)

                optimizer.zero_grad()
                cat_feature = HAM(images, net, vit_b, eye_cam)

                pred_classes = torch.max(cat_feature, dim=1)[1]
                Train_pred.extend(pred_classes.detach().cpu().tolist())

                loss = loss_function(cat_feature, labels)
                loss.backward()

                train_bar.desc = "[Fold {}][train epoch {}] lr: {:.5f}".format(
                    fold, epoch + 1, optimizer.param_groups[0]["lr"]
                )

                if not torch.isfinite(loss):
                    print('WARNING: non-finite loss, ending training ', loss)
                    sys.exit(1)

                optimizer.step()

            scheduler.step()
            print('Finished Training')

            # ----------------- Valid -----------------
            Label_val, Val_pred = [], []

            eye_prenet.eval()
            net.eval(); vit_b.eval(); HAM.eval()

            with torch.no_grad():
                val_bar = tqdm(valid_loader, file=sys.stdout)
                for step, val_data in enumerate(val_bar):
                    val_images, val_labels = val_data
                    val_images = val_images.to(device)
                    val_labels = val_labels.to(device)

                    Label_val.extend(val_labels.detach().cpu().tolist())

                    eye_cam, M, logits1 = eye_prenet(val_images)
                    cat_feature = HAM(val_images, net, vit_b, eye_cam)

                    pred_classes = torch.max(cat_feature, dim=1)[1]
                    Val_pred.extend(pred_classes.detach().cpu().tolist())

                    val_bar.desc = "[Fold {}][valid epoch {}]".format(fold, epoch + 1)


            accuracy_train, precision_train, recall_train, f1_train = compute_metrics(Label_train, Train_pred, num_classes=4)
            accuracy_val,   precision_val,   recall_val,   f1_val   = compute_metrics(Label_val,   Val_pred,   num_classes=4)

            # best checkpoint per fold
            if best_acc < accuracy_val:
                torch.save(net.state_dict(), save_path_net)
                torch.save(HAM.state_dict(), save_path_ham)
                torch.save(vit_b.state_dict(), save_path_vit)
                best_acc = accuracy_val

            print(
                '[Fold %d][epoch %d] '
                'acc_train: %.6f  prec_train: %.6f  rec_train: %.6f  f1_train: %.6f | '
                'acc_val: %.6f  prec_val: %.6f  rec_val: %.6f  f1_val: %.6f | best_val_acc: %.6f'
                % (fold, epoch + 1,
                   accuracy_train, precision_train, recall_train, f1_train,
                   accuracy_val, precision_val, recall_val, f1_val,
                   best_acc)
            )

        print(f"[Fold {fold}] Best val acc: {best_acc:.6f}")
        fold_results.append(best_acc)

    fold_results = np.array(fold_results, dtype=np.float32)
    print("\n==================== K-Fold Summary ====================")
    print("Per-fold best val acc:", fold_results.tolist())
    print("Mean ± Std:", float(fold_results.mean()), "±", float(fold_results.std(ddof=1)))


if __name__ == '__main__':
    main_kfold()