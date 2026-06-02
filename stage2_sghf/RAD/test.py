import os
import sys

import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.ham import HAM_CXR
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from custom_dataset import MedicalImageDataset,read_split_data
from sklearn.preprocessing import LabelBinarizer
from stage2_sghf.utils.tool import calculate_auc,roc_auc_score
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score



def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes, yticklabels=classes,
        ylabel='True Label',
        xlabel='Predicted Label',
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=10, fontweight='bold')
    plt.setp(ax.get_yticklabels(), fontsize=10, fontweight='bold')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)
            else:
                count = cm[i, j]
                percent = cm[i, j] / cm.sum(axis=1)[i] * 100 if cm.sum(axis=1)[i] != 0 else 0
                ax.text(j, i, f"{count}\n({percent:.1f}%)",
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=9)

    fig.tight_layout()
    plt.show()


def plot_multiclass_roc(y_true_onehot, y_score, classes, title='ROC Curve', save_path=None):

    n_classes = y_true_onehot.shape[1]
    plt.figure(figsize=(6, 5))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{classes[i]} (AUC={roc_auc:.3f})')

    fpr_micro, tpr_micro, _ = roc_curve(y_true_onehot.ravel(), y_score.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', linewidth=2,
             label=f'micro-average (AUC={roc_auc_micro:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig('roc_curve.svg', bbox_inches='tight')  # 或者.svg
        plt.show()


def plot_multiclass_pr(y_true_onehot, y_score, classes, title='Precision-Recall Curve', save_path=None):

    n_classes = y_true_onehot.shape[1]
    plt.figure(figsize=(6, 5))

    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, i], y_score[:, i])
        ap = average_precision_score(y_true_onehot[:, i], y_score[:, i])
        plt.plot(recall, precision, lw=2, label=f'{classes[i]} (AP={ap:.3f})')

    precision_micro, recall_micro, _ = precision_recall_curve(y_true_onehot.ravel(), y_score.ravel())
    ap_micro = average_precision_score(y_true_onehot, y_score, average="micro")
    plt.plot(recall_micro, precision_micro, color='gold', linestyle=':', linewidth=2,
             label=f'micro-average (AP={ap_micro:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12, fontweight='bold')
    plt.ylabel('Precision', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig('pr_curve.svg', bbox_inches='tight')  # 或者.pdf
        plt.show()
    plt.show()


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

    Batch_Size = 16

    valid_loader = DataLoader(valid_data, batch_size=Batch_Size, shuffle=False, num_workers=8, drop_last=True)

    train_num = len(train_data)
    valid_num = len(valid_data)

    print("using {} images for training, {} images for validation.".format(train_num, valid_num))

    eye_prenet = eyenet(num_classes=14,backbone='resnet101',temp=1.0,fpn_out=512,head_mid=256,use_aspp=True)
    eye_prenet_weight_path = "gaze_fpnv2_best_total.pth"
    assert os.path.exists(eye_prenet_weight_path), "file {} does not exist.".format(eye_prenet_weight_path)


    eye_prenet.load_state_dict(torch.load(eye_prenet_weight_path, map_location=device))
    print('eyenet.pth is load')
    eye_prenet.to(device)

    vit_b = vit()

    vit_b = vit_b.to(device)
    with torch.no_grad():
        _ = vit_b(torch.zeros(1, 3, 224, 224, device=device), torch.zeros(1, 1, 224, 224, device=device))  # 例如训练时用224  ,torch.zeros(1, 1, 224, 224, device=device)

    weights_path = "./vit_b.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    vit_b.load_state_dict(torch.load(weights_path, map_location=device))

    net = resnet50(stride_list=[1, 2, 1, 1], use_maxpooling=True, dilations=(1, 1, 2, 4), norm_layer=None)

    net.to(device)
    weights_path = "./res_net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))


    HAM = HAM_CXR(num_classes=4)
    HAM.to(device)
    weights_path = "./ham.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    HAM.load_state_dict(torch.load(weights_path, map_location=device))


    epochs = 1

    # conf_matrix1 = np.zeros([4,4])

    for epoch in range(epochs):

        Label_train = []
        Train_acc1 = []
        Label_val = []
        Val_acc1 = []


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
                for i in range(Batch_Size):
                    Label_val.append(labels_val[i])
                # sample_num += val_images.shape[0]

                eye_cam, M, logits1 = eye_prenet(val_images)

                cat_feature = HAM(val_images.to(device), net, vit_b, eye_cam)
                pred_classes1 = torch.max(cat_feature, dim=1)[1]
                pred_classes2 = pred_classes1.tolist()
                for i in range(Batch_Size):
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
        Label_val = np.array(Label_val)
        Val_acc1 = np.array(Val_acc1)
        num_classes = len(np.unique(Label_val))
        pred_prob = np.zeros((len(Val_acc1), num_classes))
        for i, pred in enumerate(Val_acc1):
            pred_prob[i, pred] = 1

        label_binarizer = LabelBinarizer()
        label_binarizer.fit(Label_val)
        label_val_one_hot = label_binarizer.transform(Label_val)

        auc = roc_auc_score(label_val_one_hot, pred_prob, average='macro')
        print("AUC:", auc)

        auc1 = calculate_auc(pred_prob, label_val_one_hot, 4)

        print("auc1:", auc1)

        print(
            '[epoch %d] accuracy_train: %.6f \n precision_train: %.6f \n recall_train: %.6f \n f1_train: %.6f \n accuracy_val: %.6f \n precision_val: %.6f \n recall_val: %.6f \n f1_val: %.6f' %
            (epoch + 1, accuracy_train, precision_train, recall_train, f1_train, accuracy_val, precision_val, recall_val, f1_val))

        labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        cm = confusion_matrix(Label_val, Val_acc1, labels=list(range(4)))

        fig, ax = plt.subplots(figsize=(7.2, 6.0))
        im = ax.imshow(cm, cmap=plt.cm.Blues)

        thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
        for i in range(4):
            for j in range(4):
                val = int(cm[i, j])
                ax.text(j, i, str(val),
                        ha='center', va='center',
                        color='white' if val > thresh else 'black',
                        fontsize=9)

        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10, fontweight='bold')
        ax.set_yticklabels(labels, fontsize=10, fontweight='bold')

        fig.colorbar(im, ax=ax)
        fig.tight_layout()

        plt.savefig("confusion_matrix_4cls.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

        plot_multiclass_roc(label_val_one_hot, pred_prob, labels,
                            title='ROC Curve (One-vs-Rest)',
                            save_path='roc_curve.svg')
        plot_multiclass_pr(label_val_one_hot, pred_prob, labels,
                            title='Precision-Recall Curve (One-vs-Rest)',
                            save_path='pr_curve.svg')

    print("The Best Result :")


if __name__ == '__main__':
    main()





