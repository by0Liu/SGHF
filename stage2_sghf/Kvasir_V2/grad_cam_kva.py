# ===== Grad-CAM for HAM.TripletAttention =====
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
from stage1_gazefpn.models.gazefpn import GazeFPNPlus as eyenet
from stage2_sghf.models.vit_model import vit_mws_tiny_224 as vit
from stage2_sghf.models.resnet import resnet50
from stage2_sghf.models.ham import HAM_GI
from custom_dataset import MedicalImageDataset,read_split_data

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denorm_to_uint8(img_tensor):  # img_tensor: [3,224,224], on cpu
    x = img_tensor.numpy()
    x = (x.transpose(1,2,0) * IMAGENET_STD + IMAGENET_MEAN)
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)[:, :, ::-1]  # to BGR for cv2
    return x

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def find_triplet_att_module(HAM):
    if hasattr(HAM, "TripletAttention"):
        return HAM.TripletAttention
    for name, m in HAM.named_modules():
        if "triplet" in name.lower():
            return m
    raise AttributeError("未找到 TripletAttention 模块，请确认 HAM 中的属性名，例如 self.TripletAttention")

def compute_cam_from_hooked(fmap, grads, up_size=(224,224)):

    weights = grads.mean(dim=(1,2), keepdim=False)  # [C]
    cam = torch.einsum('c,chw->hw', weights, fmap)  # [H,W]
    cam = torch.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    cam = cam.unsqueeze(0).unsqueeze(0)             # [1,1,H,W]
    cam = F.interpolate(cam, size=up_size, mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()               # [H,W], 0~1
    return cam

def overlay_cam_on_img(img_bgr_uint8, cam_01, alpha=0.6):
    heatmap = cv2.applyColorMap((cam_01 * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32)
    img = img_bgr_uint8.astype(np.float32)
    out = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return out.astype(np.uint8)

@torch.no_grad()
def _dry_run(vit_b, device):
    _ = vit_b(torch.zeros(1, 3, 224, 224, device=device))  #  , torch.zeros(1, 1, 224, 224, device=device)

def run_grad_cam_on_valid(HAM, net, vit_b, eye_prenet,
                          valid_loader, device,
                          out_dir="./gard_cam",
                          limit_batches=None,
                          target_indices=None
                          ):
    ensure_dir(out_dir)
    HAM.eval(); net.eval(); vit_b.eval(); eye_prenet.eval()

    triplet_mod = find_triplet_att_module(HAM)
    fmap_block, grad_block = [], []

    def fwd_hook(m, x, y):
        # y: Tensor [B, C, H, W]
        fmap_block.append(y.detach())

    def bwd_hook(m, gin, gout):
        # gout[0]: [B, C, H, W]
        grad_block.append(gout[0].detach())

    h1 = triplet_mod.register_forward_hook(fwd_hook)
    h2 = triplet_mod.register_full_backward_hook(bwd_hook)

    _dry_run(vit_b, device)

    sample_counter = 0
    batch_counter = 0

    for val_images, val_labels in valid_loader:
        batch_counter += 1
        if (limit_batches is not None) and (batch_counter > limit_batches):
            break

        val_images = val_images.to(device)
        val_labels = val_labels.to(device)

        with torch.no_grad():
            eye_cam, M, logits_eye = eye_prenet(val_images)  # eye_cam: [B,1,224,224] (按你的实现)

        fmap_block.clear(); grad_block.clear()
        cat_feature = HAM(val_images, net, vit_b, eye_cam)  # logits: [B, NUM_CLASSES]

        probs = torch.softmax(cat_feature, dim=1)  # [B, NUM_CLASSES]
        if target_indices is None:
            target_idx = probs.argmax(dim=1)       # [B]
        else:
            target_idx = torch.as_tensor(target_indices[:probs.size(0)], device=device)

        for b in range(val_images.size(0)):
            HAM.zero_grad(set_to_none=True); net.zero_grad(set_to_none=True); vit_b.zero_grad(set_to_none=True)

            fmap_block.clear(); grad_block.clear()
            logits_b = HAM(val_images[b:b+1], net, vit_b, eye_cam[b:b+1])  # [1, NUM_CLASSES]

            cls = target_idx[b].item()
            score = logits_b[0, cls]
            score.backward(retain_graph=False)

            assert len(fmap_block) == 1 and len(grad_block) == 1, "Grad-CAM 钩子未捕获到 fmap/grad，请检查模块名"

            fmap = fmap_block[0].squeeze(0)  # [C,H,W]
            grads = grad_block[0].squeeze(0) # [C,H,W]

            cam = compute_cam_from_hooked(fmap, grads, up_size=(224,224))

            img_uint8 = denorm_to_uint8(val_images[b].detach().cpu())
            vis = overlay_cam_on_img(img_uint8, cam, alpha=0.6)

            true_cls = int(val_labels[b].item())
            pred_cls = int(cls)

            save_cam = os.path.join(out_dir, f"idx{sample_counter:06d}_true{true_cls}_pred{pred_cls}_noDBAT.jpg")
            save_raw = os.path.join(out_dir, f"idx{sample_counter:06d}_true{true_cls}_pred{pred_cls}_raw_nogaze.jpg")

            cv2.imwrite(save_cam, vis)

            sample_counter += 1

    # 解绑钩子
    h1.remove(); h2.remove()
    print(f"[Grad-CAM] Saved {sample_counter} visualizations to: {out_dir}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform_valid = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    data_path = "./dataset/kvasir-v2"
    _, _, val_images_path, val_images_label = read_split_data(data_path)
    valid_data = MedicalImageDataset(val_images_path, val_images_label, transform=transform_valid)
    valid_loader = DataLoader(valid_data, batch_size=4, shuffle=False)

    eye_prenet = eyenet(num_classes=3, backbone='resnet101', temp=1.0, fpn_out=512,
                        head_mid=256, use_aspp=True).to(device)
    eye_prenet.load_state_dict(torch.load("gaze_fpnv2_polyp_best_total.pth", map_location=device))

    vit_b = vit().to(device)
    with torch.no_grad():
        _ = vit_b(torch.zeros(1, 3, 224, 224, device=device), torch.zeros(1, 1, 224, 224, device=device))  # 例如训练时用224,  torch.zeros(1, 1, 224, 224, device=device)
    vit_b.load_state_dict(torch.load("./vit_b.pth", map_location=device))

    net = resnet50(stride_list=[1, 2, 1, 1],
                   use_maxpooling=True,
                   dilations=(1, 1, 2, 4),
                   norm_layer=None).to(device)
    net.load_state_dict(torch.load("./res_net.pth", map_location=device))

    HAM = HAM_GI(num_classes=8).to(device)
    HAM.load_state_dict(torch.load("./ham.pth", map_location=device))

    run_grad_cam_on_valid(
        HAM=HAM,
        net=net,
        vit_b=vit_b,
        eye_prenet=eye_prenet,
        valid_loader=valid_loader,
        device=device,
        out_dir="./gard_cam",
        limit_batches=150
    )

if __name__ == '__main__':
    main()


