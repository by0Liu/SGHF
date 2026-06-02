# ===== Grad-CAM (HAM.TripletAttention) + BBox overlay for NIH14 =====
import os
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.resnet import resnet50
from stage2_sghf.models.ham import HAM_CXR_14
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from custom_dataset import NIHDataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)
def _safe_name(s):  return s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")

def _denorm_to_bgr_uint8(x_3hw):
    x = x_3hw.detach().cpu().float().numpy()
    x = (np.transpose(x, (1,2,0)) * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    x = x[:, :, ::-1].copy()
    return x[:, :, ::-1]  # RGB->BGR

def _find_triplet_module(HAM):
    if hasattr(HAM, "TripletAttention"): return HAM.TripletAttention
    for name, m in HAM.named_modules():
        if "triplet" in name.lower():
            return m
    raise AttributeError("未找到 TripletAttention 模块，请检查 HAM 的属性名")

def _compute_cam(fmap_chw, grads_chw, up_hw=(224,224)):
    weights = grads_chw.mean(dim=(1,2))
    cam = torch.einsum('c,chw->hw', weights, fmap_chw)
    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0: cam /= cam.max()
    cam = F.interpolate(cam[None,None,...], size=up_hw, mode='bilinear', align_corners=False)
    return cam[0,0].detach().cpu().numpy()

def _overlay(img_bgr_uint8, cam01, alpha=0.6):
    heat = cv2.applyColorMap((cam01*255).astype(np.uint8), cv2.COLORMAP_JET)
    out  = cv2.addWeighted(heat, alpha, img_bgr_uint8, 1-alpha, 0)
    return np.ascontiguousarray(out)

def _names_from_multilabel(vec01, class_names):
    idx = np.where(vec01 > 0.5)[0].tolist()
    return "None" if not idx else "+".join([class_names[i] for i in idx])

def load_box_file(box_txt_path):
    mapping = {}
    with open(box_txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            name = os.path.basename(parts[0])
            try:
                x, y, w, h = map(float, parts[-4:])
                mapping[name] = (x, y, w, h)
            except:
                pass
    return mapping


def scale_box_xywh(box, src_size, dst_size):
    sx = float(dst_size) / float(src_size)
    x, y, w, h = box
    return (x*sx, y*sx, w*sx, h*sx)

def draw_box(img_bgr, box_xywh, color=(0,255,255), thickness=3):
    img_bgr = np.ascontiguousarray(img_bgr)
    x,y,w,h = box_xywh
    p1 = (int(round(x)), int(round(y)))
    p2 = (int(round(x+w)), int(round(y+h)))
    cv2.rectangle(img_bgr, p1, p2, color, thickness)

def run_grad_cam_on_test_NIH14_with_boxes(
    HAM, net, vit_b, eye_prenet, test_loader, device,
    class_names,
    out_dir="gradcam_triplet_nih14",
    up_hw=(224,224),
    target="pred_top1",
    limit_batches=None,
    box_txt_path=None,
    get_image_name=None,
    box_src_size=1024,
    box_dst_size=None
):

    _ensure_dir(out_dir)
    HAM.eval(); net.eval(); vit_b.eval(); eye_prenet.eval()

    box_map = load_box_file(box_txt_path) if box_txt_path else {}
    if box_dst_size is None:
        box_dst_size = max(up_hw)

    target_mod = _find_triplet_module(HAM)
    fmap_block, grad_block = [], []
    def fwd_hook(m, x, y):  fmap_block.append(y)
    def bwd_hook(m, gin, gout): grad_block.append(gout[0])
    h1 = target_mod.register_forward_hook(fwd_hook)
    h2 = target_mod.register_full_backward_hook(bwd_hook)

    with torch.no_grad():
        _ = vit_b(torch.zeros(1,3,up_hw[0],up_hw[1],device=device))  #torch.zeros(1,1,up_hw[0],up_hw[1],device=device)

    sample_idx = 0
    seen = 0
    dataset = test_loader.dataset

    for bi, (images, labels) in enumerate(test_loader):
        if (limit_batches is not None) and (bi >= limit_batches): break

        B = images.size(0)
        images = images.to(device)
        labels = labels.to(device).float()

        with torch.no_grad():
            eye_cam, M, _ = eye_prenet(images)

        with torch.enable_grad():
            fmap_block.clear(); grad_block.clear()
            logits = HAM(images, net, vit_b, eye_cam)  # [B,K]
            probs  = torch.sigmoid(logits)

        for b in range(B):
            img_u8 = _denorm_to_bgr_uint8(images[b])
            gt_vec = labels[b].detach().cpu().numpy()
            gt_name = _safe_name(_names_from_multilabel(gt_vec, class_names))

            global_idx = seen + b
            img_name = get_image_name(dataset, global_idx) if callable(get_image_name) else None

            if (not img_name) or (os.path.basename(img_name) not in box_map):
                continue

            box_xywh_dst = scale_box_xywh(box_map[os.path.basename(img_name)],
                                          src_size=box_src_size,
                                          dst_size=box_dst_size)

            if img_name and (img_name in box_map):
                box_xywh_dst = scale_box_xywh(box_map[img_name], src_size=box_src_size, dst_size=box_dst_size)

            if target == "pred_top1":
                cls = int(torch.argmax(probs[b]).item())
                net.zero_grad(True); vit_b.zero_grad(True); HAM.zero_grad(True)
                fmap_block.clear(); grad_block.clear()
                logits_b = HAM(images[b:b+1], net, vit_b, eye_cam[b:b+1])
                score = logits_b[0, cls]
                score.backward()

                assert len(fmap_block)==1 and len(grad_block)==1, "钩子未捕获 fmap/grad，请检查模块名"
                fmap = fmap_block[0].squeeze(0)
                grad = grad_block[0].squeeze(0)
                cam  = _compute_cam(fmap, grad, up_hw=up_hw)
                vis  = _overlay(img_u8, cam, alpha=0.6)

                if box_xywh_dst is not None:
                    sx = up_hw[1] / float(box_dst_size)
                    sy = up_hw[0] / float(box_dst_size)
                    x,y,w,h = box_xywh_dst
                    box_scaled = (x*sx, y*sy, w*sx, h*sy)
                    draw_box(vis, box_scaled, color=(0,255,255), thickness=3)
                    draw_box(img_u8, box_scaled, color=(0,255,255), thickness=3)

                pred_name = _safe_name(class_names[cls])
                fn_cam = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_pred[{pred_name}]_nogaze.jpg")
                fn_raw = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_pred[{pred_name}]_raw.jpg")
                cv2.imwrite(fn_cam, vis)
                # cv2.imwrite(fn_raw, img_u8)
                sample_idx += 1

            elif target == "gt_each":
                pos_idx = torch.where(labels[b] > 0.5)[0].tolist()
                if not pos_idx:
                    pos_idx = [int(torch.argmax(probs[b]).item())]
                for cls in pos_idx:
                    net.zero_grad(True); vit_b.zero_grad(True); HAM.zero_grad(True)
                    fmap_block.clear(); grad_block.clear()
                    logits_b = HAM(images[b:b+1], net, vit_b, eye_cam[b:b+1])
                    score = logits_b[0, int(cls)]
                    score.backward()

                    assert len(fmap_block)==1 and len(grad_block)==1, "钩子未捕获 fmap/grad，请检查模块名"
                    fmap = fmap_block[0].squeeze(0)
                    grad = grad_block[0].squeeze(0)
                    cam  = _compute_cam(fmap, grad, up_hw=up_hw)
                    vis  = _overlay(img_u8, cam, alpha=0.6)

                    if box_xywh_dst is not None:
                        sx = up_hw[1] / float(box_dst_size)
                        sy = up_hw[0] / float(box_dst_size)
                        x,y,w,h = box_xywh_dst
                        box_scaled = (x*sx, y*sy, w*sx, h*sy)
                        draw_box(vis, box_scaled, color=(0,255,255), thickness=3)
                        draw_box(img_u8, box_scaled, color=(0,255,255), thickness=3)

                    cls_name = _safe_name(class_names[int(cls)])
                    fn_cam = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_gt[{cls_name}]_nogaze.jpg")
                    fn_raw = os.path.join(out_dir, f"idx{sample_idx:06d}_true[{gt_name}]_gt[{cls_name}]_raw.jpg")
                    cv2.imwrite(fn_cam, vis)
                    # cv2.imwrite(fn_raw, img_u8)
                    sample_idx += 1
            else:
                raise ValueError("target 仅支持 'pred_top1' 或 'gt_each'。")

        seen += B

    h1.remove(); h2.remove()
    print(f"[Grad-CAM] saved to: {out_dir}")

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

    box_txt = os.path.join(data_root, "csv", "box.txt")


    def get_name_from_dataset(dataset, global_idx):
        return os.path.basename(dataset.image_names[global_idx])

    run_grad_cam_on_test_NIH14_with_boxes(
        HAM=HAM, net=net, vit_b=vit_b, eye_prenet=eye_prenet,
        test_loader=test_loader, device=device,
        class_names=Class_Names,
        out_dir="./gradcam_nih14_boxes",
        up_hw=(224, 224),
        target="pred_top1",
        limit_batches=1402,
        box_txt_path=box_txt,
        get_image_name=get_name_from_dataset,
        box_src_size=1024,
        box_dst_size=224
    )

# cv2.imwrite(fn_raw, img_u8)

if __name__ == '__main__':
    main()
