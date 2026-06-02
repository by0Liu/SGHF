import sys
import math
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.ham import HAM_CXR_14
from stage2_sghf.utils.tool import compute_AUCs
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from custom_dataset import NIHDataset
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from pathlib import Path
from auc_vs_k import mean_auc_vs_label_cardinality, plot_mean_auc_vs_k, save_mean_auc_vs_k_csv

import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



Class_Names = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
           'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

ABBR = {
    'Atelectasis':'Atel','Cardiomegaly':'Card','Effusion':'Effu','Infiltration':'Infi',
    'Mass':'Mass','Nodule':'Nodu','Pneumonia':'Pneu','Pneumothorax':'Pneu2',
    'Consolidation':'Cons','Edema':'Edem','Emphysema':'Emph','Fibrosis':'Fibr',
    'Pleural_Thickening':'P_T','Hernia':'Hern'
}
def abbr(name): return ABBR.get(name, name[:4])

def _ensure_prob(y_score_np):
    if y_score_np.min() < 0.0 or y_score_np.max() > 1.0:
        y_score_np = 1.0 / (1.0 + np.exp(-y_score_np))
    return y_score_np

def plot_multilabel_roc_pr(y_true_t, y_score_t, class_names, out_dir="./figs", prefix="nih14"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    y_true = y_true_t.detach().float().cpu().numpy()
    y_score = _ensure_prob(y_score_t.detach().float().cpu().numpy())
    n_classes = y_true.shape[1]

    # ===== ROC per-class / micro / macro =====
    fpr, tpr, roc_auc = {}, {}, {}
    for c in range(n_classes):
        if np.sum(y_true[:, c]) == 0 or np.sum(1 - y_true[:, c]) == 0:
            fpr[c] = np.array([0, 1]); tpr[c] = np.array([0, 1]); roc_auc[c] = np.nan
            continue
        fpr[c], tpr[c], _ = roc_curve(y_true[:, c], y_score[:, c])
        roc_auc[c] = auc(fpr[c], tpr[c])

    plt.figure(figsize=(7, 5))
    for c, name in enumerate(class_names):
        if np.isnan(roc_auc[c]): continue
        plt.plot(fpr[c], tpr[c], lw=1, label=f"{abbr(name)} {roc_auc[c]:.3f}")
    plt.plot([0, 1], [0, 1], '--', lw=1)
    mean_auc = np.nanmean([roc_auc[c] for c in range(n_classes)])
    plt.title(f"AVG AUC={mean_auc:.3f}")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"{prefix}_roc.png", dpi=300); plt.close()

    # ===== PR per-class / micro =====
    precision, recall, ap = {}, {}, {}
    for c in range(n_classes):
        if np.sum(y_true[:, c]) == 0:
            precision[c] = np.array([1.0]); recall[c] = np.array([0.0]); ap[c] = np.nan
            continue
        precision[c], recall[c], _ = precision_recall_curve(y_true[:, c], y_score[:, c])
        ap[c] = average_precision_score(y_true[:, c], y_score[:, c])

    # precision["micro"], recall["micro"], _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
    # ap["micro"] = average_precision_score(y_true, y_score, average="micro")
    # ap_macro = np.nanmean([ap[c] for c in range(n_classes)])  # macro AP

    base_pos_rate = y_true.mean()
    plt.figure(figsize=(7, 5))
    plt.plot([0, 1], [base_pos_rate, base_pos_rate], '--', lw=1)
    for c, name in enumerate(class_names):
        if np.isnan(ap[c]): continue
        plt.plot(recall[c], precision[c], lw=1, label=f"{abbr(name)} {ap[c]:.3f}")
    # plt.plot(recall["micro"], precision["micro"], lw=2, label=f"micro {ap['micro']:.3f}")
    mean_ap = np.nanmean([ap[c] for c in range(n_classes)])
    plt.title(f"AVG AP={mean_ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(Path(out_dir) / f"{prefix}_pr.png", dpi=300); plt.close()


    metrics = {name: {"AUC": (None if np.isnan(roc_auc[i]) else float(roc_auc[i])),
                      "AP":  (None if np.isnan(ap[i]) else float(ap[i]))}
               for i, name in enumerate(class_names)}
    # metrics["_micro"] = {"AUC": float(roc_auc["micro"]), "AP": float(ap["micro"])}
    # metrics["_macro"] = {"AUC": float(roc_auc["macro"]), "AP": float(ap_macro)}
    return metrics



# ===== Empirical Analysis Utils =====

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_numpy_img(batch_tensor):
    x = batch_tensor.detach().cpu().float().numpy()  # B,3,H,W in [normalized]
    x = (np.transpose(x, (0,2,3,1)) * IMAGENET_STD) + IMAGENET_MEAN  # B,H,W,3
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return [x[i] for i in range(x.shape[0])]

def _format_topk_for_one(prob_vec, y_true_vec, class_names, topk=8):
    prob = prob_vec
    gt = y_true_vec.astype(np.int32)
    order = np.argsort(-prob)  # desc
    topk_idx = order[:topk]
    lines_pred = []
    for j in topk_idx:
        mark = "*" if gt[j] == 1 else ""
        lines_pred.append(f"{mark}{class_names[j]}: {prob[j]:.3f}")
    missed_idx = np.where(gt == 1)[0]
    missed_idx = [j for j in missed_idx if j not in topk_idx]
    missed_gt = [class_names[j] for j in missed_idx]
    return topk_idx, lines_pred, missed_gt

def save_empirical_panel(images_bchw, probs_bk, ytrue_bk, class_names, out_path, topk=8):
    imgs = _to_numpy_img(images_bchw)  # list of HxWx3
    B = len(imgs)
    K = probs_bk.shape[1]

    cols = int(math.ceil(math.sqrt(B)))
    rows = int(math.ceil(B / cols))

    plt.figure(figsize=(4.2*cols, 3.8*rows), dpi=150)
    for i in range(B):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(imgs[i])
        ax.axis('off')

        topk_idx, lines_pred, missed_gt = _format_topk_for_one(
            probs_bk[i], ytrue_bk[i], class_names, topk=topk
        )

        y0 = 0.03
        dy = 0.065
        ax.text(0.02, y0, "Top-8:", transform=ax.transAxes, fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.5, alpha=0.7))
        for li, line in enumerate(lines_pred):
            color = "tab:blue" if line.startswith("*") else "black"
            line2 = line[1:] if line.startswith("*") else line
            ax.text(0.02, y0 + (li+1)*dy, line2, transform=ax.transAxes, fontsize=8, color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.55))

        if len(missed_gt) > 0:
            msg = "Missed: " + ", ".join(missed_gt[:4])
            ax.text(0.02, y0 + (len(lines_pred)+1)*dy, msg, transform=ax.transAxes, fontsize=8, color="tab:red",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.55))

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()



def main():

    TARGET_BATCH_IDX = 1
    TOPK = 8

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

    test_loader = DataLoader(test_data, batch_size=Batch_Size, shuffle=False, num_workers=8,drop_last = True)

    train_num = len(train_data)
    valid_num = len(valid_data)

    # print("using {} images for training, {} images for validation, {} images for testing.".format(train_num,valid_num,test_num))
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
        _ = vit_b(torch.zeros(1, 3, 224, 224, device=device),torch.zeros(1, 1, 224, 224, device=device))  # 例如训练时用224  ,torch.zeros(1, 1, 224, 224, device=device)

    weights_path = "./vit_b.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    vit_b.load_state_dict(torch.load(weights_path, map_location=device))

    net = resnet50(stride_list=[1, 2, 1, 1], use_maxpooling=True, dilations=(1, 1, 2, 4), norm_layer=None)

    net.to(device)
    weights_path = "./res_net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    net.load_state_dict(torch.load(weights_path, map_location=device))

    HAM = HAM_CXR_14(num_classes=14)
    HAM.to(device)
    weights_path = "./ham.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    HAM.load_state_dict(torch.load(weights_path, map_location=device))

    epochs = 1

    conf_matrix1 = np.zeros([4,4])

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

        test_auc = []

        gt_test = torch.FloatTensor()
        gt_test = gt_test.to(device)
        pred_test = torch.FloatTensor()
        pred_test = pred_test.to(device)

        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for step, test_data in enumerate(test_bar):
                test_images, test_labels = test_data
                test_images = test_images.to(device)
                test_labels = test_labels.to(device)
                gt_test = torch.cat((gt_test, test_labels), 0)

                labels_val = test_labels.tolist()
                for i in range(Batch_Size):
                    Label_val.append(labels_val[i])
                # sample_num += val_images.shape[0]


                eye_cam, M, logits1 = eye_prenet(test_images)
                cat_feature = HAM(test_images, net, vit_b, eye_cam)

                # ===== Empirical analysis for one batch =====
                if step == TARGET_BATCH_IDX:
                    probs_batch = torch.sigmoid(cat_feature).detach().cpu().numpy()   # (B,14)
                    ytrue_batch = test_labels.detach().cpu().numpy().astype(np.int32) # (B,14)
                    out_png = f"./figs/empirical_nih14_HAM_b{TARGET_BATCH_IDX}.png"
                    save_empirical_panel(
                        images_bchw=test_images,
                        probs_bk=probs_batch,
                        ytrue_bk=ytrue_batch,
                        class_names=Class_Names,
                        out_path=out_png,
                        topk=TOPK
                    )
                    for bi in range(probs_batch.shape[0]):
                        order = np.argsort(-probs_batch[bi])
                        top_idx = order[:TOPK]
                        top_pairs = [f"{Class_Names[j]}={probs_batch[bi][j]:.3f}" for j in top_idx]
                        gt_pos = [Class_Names[j] for j in np.where(ytrue_batch[bi] == 1)[0]]
                        missed = [g for g in gt_pos if g not in [Class_Names[j] for j in top_idx]]
                        print(f"[Batch {step} | Img {bi}] Top-{TOPK}: {top_pairs}")
                        if len(missed) > 0:
                            print(f"    Missed GT: {missed}")
                        else:
                            print(f"    Missed GT: None")
                # ===== end empirical analysis =====

                pred_test = torch.cat((pred_test, cat_feature.data), 0)
                # print(pred_test)

        AUROCs_test = compute_AUCs(gt_test, pred_test)
        AUROC_avg_test = np.array(AUROCs_test).mean()

        test_auc.append(AUROC_avg_test.item())
        with open("./test_mean_auc.txt", 'w') as test_ac:
            test_ac.write(str(test_auc))

        for i in range(14):
            print('The AUROC_test of {} is {}'.format(Class_Names[i], AUROCs_test[i]))

        print('test_mean_auc: %.3f' %
                (AUROC_avg_test))

        metrics = plot_multilabel_roc_pr(gt_test, pred_test, Class_Names, out_dir="./figs", prefix="nih14_HAM")
        for k, v in metrics.items():
            print(k, v)


        # ====== Label Cardinality Stratification: Mean AUC vs k ======
        y_true_np  = gt_test.detach().float().cpu().numpy()
        y_score_np = pred_test.detach().float().cpu().numpy()

        ks_labels, mean_auc_per_bin, counts_per_bin = mean_auc_vs_label_cardinality(
            y_true=y_true_np,
            y_score=y_score_np,
            bins=(1, 2, 3, 99)
        )

        plot_mean_auc_vs_k(
            ks_labels, mean_auc_per_bin, counts_per_bin,
            out_png="./figs/nih14_mean_auc_vs_k.png",
            title=r"Mean AUC vs Label Cardinality ($k$)"
        )
        save_mean_auc_vs_k_csv(
            ks_labels, mean_auc_per_bin, counts_per_bin,
            out_csv="./figs/nih14_mean_auc_vs_k.csv"
        )

        print("[Mean AUC vs k] bins:", ks_labels)
        print("[Mean AUC vs k] values:", mean_auc_per_bin)
        print("[Mean AUC vs k] counts:", counts_per_bin)
        # ====== end stratification ======




if __name__ == '__main__':
    main()





