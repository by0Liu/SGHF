# ===== Grad-CAM for HAM.TripletAttention on NIH14 (multi-label) =====
import os
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.ham import HAM_CXR_14
from stage2_sghf.models.resnet import resnet50
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from custom_dataset import NIHDataset


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _denorm_to_bgr_uint8(x_3hw):
    """ x_3hw: torch.Tensor [3,H,W] (normalized) -> np.uint8 HxWx3 (BGR) """
    x = x_3hw.detach().cpu().float().numpy()
    x = (np.transpose(x, (1,2,0)) * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    return x[:, :, ::-1]  # RGB->BGR for cv2

def _find_triplet_module(HAM):
    if hasattr(HAM, "TripletAttention"):
        return HAM.TripletAttention
    for name, m in HAM.named_modules():
        if "triplet" in name.lower():
            return m
    raise AttributeError("未找到 TripletAttention 模块，请确认 HAM 内部的属性名。")

def _compute_cam(fmap_chw, grads_chw, up_hw=(224,224)):
    """ fmap, grads: torch.Tensor [C,H,W] """
    weights = grads_chw.mean(dim=(1,2))                   # [C]
    cam = torch.einsum('c,chw->hw', weights, fmap_chw)    # [H,W]
    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam = F.interpolate(cam[None,None,...], size=up_hw, mode='bilinear', align_corners=False)
    return cam[0,0].detach().cpu().numpy()  # [H,W] 0~1

def _overlay(img_bgr_uint8, cam01, alpha=0.6):
    heat = cv2.applyColorMap((cam01*255).astype(np.uint8), cv2.COLORMAP_JET)
    out  = cv2.addWeighted(heat.astype(np.uint8), alpha, img_bgr_uint8, 1-alpha, 0)
    return out

def _names_from_multilabel(vec01, class_names):
    """ vec01: np.array [K] of 0/1 -> 'ClassA+ClassB' 或 'None' """
    idx = np.where(vec01 > 0.5)[0].tolist()
    if not idx: return "None"
    return "+".join([class_names[i] for i in idx])

def _safe_name(s):
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

def run_grad_cam_on_test_NIH14(
    HAM, net, vit_b, eye_prenet, test_loader, device,
    class_names,
    out_dir="gradcam_triplet_nih14",
    target="pred_top1",
    limit_batches=None
):

    _ensure_dir(out_dir)
    HAM.eval(); net.eval(); vit_b.eval(); eye_prenet.eval()

    target_mod = _find_triplet_module(HAM)
    fmap_block, grad_block = [], []
    def fwd_hook(m, x, y):  # y: [B,C,H,W]
        fmap_block.append(y)
    def bwd_hook(m, gin, gout):  # gout[0]: [B,C,H,W]
        grad_block.append(gout[0])
    h1 = target_mod.register_forward_hook(fwd_hook)
    h2 = target_mod.register_full_backward_hook(bwd_hook)

    with torch.no_grad():
        _ = vit_b(torch.zeros(1,3,224,224,device=device))

    sample_idx = 0
    for bi, (images, labels) in enumerate(test_loader):
        if (limit_batches is not None) and (bi >= limit_batches): break

        images = images.to(device)         # [B,3,224,224]
        labels = labels.to(device).float() # [B,14] 0/1

        with torch.no_grad():
            eye_cam, M, logits_eye = eye_prenet(images)

        B = images.size(0)
        with torch.enable_grad():
            fmap_block.clear(); grad_block.clear()
            logits = HAM(images, net, vit_b, eye_cam)    # [B,K]
            probs  = torch.sigmoid(logits)               # [B,K]

        for b in range(B):
            img_u8 = _denorm_to_bgr_uint8(images[b])

            gt_vec = labels[b].detach().cpu().numpy()    # [K] 0/1
            gt_name = _names_from_multilabel(gt_vec, class_names)
            gt_name = _safe_name(gt_name)

            if target == "pred_top1":
                cls = int(torch.argmax(probs[b]).item())
                net.zero_grad(set_to_none=True); vit_b.zero_grad(set_to_none=True); HAM.zero_grad(set_to_none=True)
                fmap_block.clear(); grad_block.clear()

                logits_b = HAM(images[b:b+1], net, vit_b, eye_cam[b:b+1])  # [1,K]
                score = logits_b[0, cls]
                score.backward()

                assert len(fmap_block)==1 and len(grad_block)==1, "钩子未捕获 fmap/grad，请检查模块名。"
                fmap = fmap_block[0].squeeze(0)  # [C,H,W]
                grad = grad_block[0].squeeze(0)  # [C,H,W]
                cam  = _compute_cam(fmap, grad, up_hw=(224,224))
                vis  = _overlay(img_u8, cam, alpha=0.6)

                pred_name = _safe_name(class_names[cls])
                fn_cam = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_pred[{pred_name}]_cam.jpg")
                fn_raw = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_pred[{pred_name}]_raw.jpg")
                cv2.imwrite(fn_cam, vis)
                cv2.imwrite(fn_raw, img_u8)
                sample_idx += 1

            elif target == "gt_each":
                pos_idx = torch.where(labels[b] > 0.5)[0].tolist()
                if not pos_idx:
                    cls = int(torch.argmax(probs[b]).item())
                    pos_idx = [cls]
                for cls in pos_idx:
                    net.zero_grad(set_to_none=True); vit_b.zero_grad(set_to_none=True); HAM.zero_grad(set_to_none=True)
                    fmap_block.clear(); grad_block.clear()

                    logits_b = HAM(images[b:b+1], net, vit_b, eye_cam[b:b+1])
                    score = logits_b[0, int(cls)]
                    score.backward()

                    assert len(fmap_block)==1 and len(grad_block)==1, "钩子未捕获 fmap/grad，请检查模块名。"
                    fmap = fmap_block[0].squeeze(0)
                    grad = grad_block[0].squeeze(0)
                    cam  = _compute_cam(fmap, grad, up_hw=(224,224))
                    vis  = _overlay(img_u8, cam, alpha=0.6)

                    cls_name = _safe_name(class_names[int(cls)])
                    fn_cam = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_gt[{cls_name}]_cam.jpg")
                    fn_raw = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_gt[{cls_name}]_raw.jpg")
                    cv2.imwrite(fn_cam, vis)
                    cv2.imwrite(fn_raw, img_u8)
                    sample_idx += 1
            else:
                raise ValueError("target 仅支持 'pred_top1' 或 'gt_each'。")

    h1.remove(); h2.remove()
    print(f"[Grad-CAM] saved images to: {out_dir}")


# gradcam_run.py

def main():

    torch.manual_seed(3407)
    np.random.seed(3407)
    torch.cuda.manual_seed_all(3407)

    Class_Names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                   'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    transform_test = transforms.Compose([transforms.Resize(224),
                                         # transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    data_root = os.path.abspath(os.path.join("/media/user/0291ebd7-713f-4d1c-9a78-ac6224af6b10/home/cvnlp/archive/chext dataset"))
    image_path = os.path.join(data_root, 'images')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    test_list_path = os.path.join(data_root, 'csv', 'test_list.txt')
    assert os.path.exists(test_list_path), "{} path does not exist.".format(test_list_path)
    test_data = NIHDataset(data_path=image_path, image_list_path=test_list_path, transform=transform_test)
    Batch_Size = 16
    test_loader = DataLoader(test_data, batch_size=Batch_Size, shuffle=False, num_workers=8,drop_last = True)

    eye_prenet = eyenet(num_classes=14, backbone='resnet101', temp=1.0, fpn_out=512, head_mid=256, use_aspp=True)
    eye_prenet_weight_path = "gaze_fpnv2_best_total.pth"
    assert os.path.exists(eye_prenet_weight_path), "file {} does not exist.".format(eye_prenet_weight_path)
    eye_prenet.load_state_dict(torch.load(eye_prenet_weight_path, map_location=device))
    eye_prenet.to(device)

    vit_b = vit()
    vit_b = vit_b.to(device)
    with torch.no_grad():
        _ = vit_b(torch.zeros(1, 3, 224, 224, device=device))


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



    run_grad_cam_on_test_NIH14(
        HAM=HAM,
        net=net,
        vit_b=vit_b,
        eye_prenet=eye_prenet,
        test_loader=test_loader,
        device=device,
        class_names=Class_Names,
        out_dir="./gradcam_nih14",
        target="pred_top1",
        limit_batches=1
    )

if __name__ == '__main__':
    main()
