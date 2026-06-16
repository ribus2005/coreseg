import os

import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
import torch.nn.functional as F

import cv2
import torch
import torch.nn as nn

from sklearn.metrics import precision_recall_curve, auc

import threading
import queue
    

class Decoder(nn.Module):
    def __init__(self, in_channels=256, num_classes=21, upscale_factor = 32):
        super().__init__()
        assert upscale_factor == 32 or upscale_factor == 16
        # 1. H/32 -> H/16
        in_ch = in_channels
        self.deconv1 = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(in_ch // 2)
        
        # 2. H/16 -> H/8
        self.deconv2 = nn.ConvTranspose2d(in_ch // 2, in_ch // 4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(in_ch // 4)
        
        # 3. H/8 -> H/4
        self.deconv3 = nn.ConvTranspose2d(in_ch // 4, in_ch // 8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(in_ch // 8)
        

        if upscale_factor == 32:
            # 4. H/4 -> H / 2
            self.deconv4 = nn.ConvTranspose2d(in_ch // 8, in_ch // 16, kernel_size=4, stride=2, padding=1)
            self.bn4 = nn.BatchNorm2d(in_ch // 16)

            self.deconv5 = nn.ConvTranspose2d(in_ch // 16, num_classes, kernel_size=4, stride=2, padding=1)
        elif upscale_factor == 16:
            self.deconv4 = nn.ConvTranspose2d(in_ch // 8, num_classes, kernel_size=4, stride=2, padding=1)
            self.bn4 = nn.Identity()
            self.deconv5 = nn.Identity()

        self.num_classes = num_classes

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = F.relu(self.bn4(self.deconv4(x)))
        x = self.deconv5(x)  # logits без активации
        return x
    
    def reset(self):
            for m in self.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()


def compute_miou(predictions, targets, num_classes):
    if predictions.shape[1] > 1:
        preds = predictions.argmax(dim=1)  # [B, H, W]
    else:
        preds = predictions.squeeze()
    
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        target_cls = (targets == cls)
        inter = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union > 0:
            ious.append(inter / union)
    
    return sum(ious) / len(ious) if ious else 0.0


def crop_4(image: np.ndarray, crop_size: int = 512):
    """
    Делит одно изображение (NumPy array) на 4 кропа заданного размера.

    Args:
        image (np.ndarray): массив [H, W] или [H, W, C]
        crop_size (int): размер кропа

    Returns:
        patches (list of np.ndarray): список из 4 кропов
    """
    if image.ndim == 2:
        H, W = image.shape
        C = None
    elif image.ndim == 3:
        H, W, C = image.shape
    else:
        raise ValueError("image должен быть [H, W] или [H, W, C]")

    if H < crop_size or W < crop_size:
        raise ValueError(f"Изображение слишком маленькое ({H}x{W}) для кропа {crop_size}x{crop_size}")

    # верхний левый
    patch1 = image[0:crop_size, 0:crop_size] if C is None else image[0:crop_size, 0:crop_size, :]
    # верхний правый
    patch2 = image[0:crop_size, W-crop_size:W] if C is None else image[0:crop_size, W-crop_size:W, :]
    # нижний левый
    patch3 = image[H-crop_size:H, 0:crop_size] if C is None else image[H-crop_size:H, 0:crop_size, :]
    # нижний правый
    patch4 = image[H-crop_size:H, W-crop_size:W] if C is None else image[H-crop_size:H, W-crop_size:W, :]

    return [patch1, patch2, patch3, patch4]


def plot_pca_2d(pca_2d_embeddings, labels, epoch=None, title=None, 
                connect_pairs=False, view_markers=None):
    """
    Визуализация PCA [N, 2] с цветными метками
    
    Args:
        pca_2d_embeddings: np.array [N, 2] — координаты после PCA
        labels: list/array [N] — метки (например, ID изображений)
        connect_pairs: bool — соединять ли линии между view1/view2 одного изображения
        view_markers: dict — {'v1': 'o', 'v2': 'x'} для разных маркеров
    """
    plt.figure(figsize=(8, 6))
    
    # Цветовая схема: каждая уникальная метка — свой цвет
    unique_labels = np.unique(labels)
    color_map = plt.get_cmap('tab10', len(unique_labels))
    label_to_color = {lbl: color_map(i) for i, lbl in enumerate(unique_labels)}
    
    # Рисуем точки
    for i, (x, y) in enumerate(pca_2d_embeddings):
        lbl = labels[i]
        marker = 'o'
        
        # Если есть информация о view (например, '42_v1', '42_v2')
        if view_markers and isinstance(lbl, str):
            for view_key, marker_symbol in view_markers.items():
                if view_key in lbl:
                    marker = marker_symbol
                    break
        
        plt.scatter(x, y, c=[label_to_color[lbl]], marker=marker, 
                   s=40, alpha=0.7, edgecolors='white', linewidth=0.5)
    
    # Соединяем пары одного изображения (для DINO: view1 ↔ view2)
    if connect_pairs:
        # Ожидаем метки вида '42_v1', '42_v2'
        img_to_idx = {}
        for i, lbl in enumerate(labels):
            if isinstance(lbl, str) and '_' in lbl:
                img_id = lbl.rsplit('_', 1)[0]  # '42_v1' → '42'
                if img_id not in img_to_idx:
                    img_to_idx[img_id] = []
                img_to_idx[img_id].append(i)
        
        for img_id, idx_list in img_to_idx.items():
            if len(idx_list) == 2:  # ровно 2 views
                i1, i2 = idx_list
                x1, y1 = pca_2d_embeddings[i1]
                x2, y2 = pca_2d_embeddings[i2]
                plt.plot([x1, x2], [y1, y2], 'gray', linewidth=0.8, alpha=0.3)
    
    # Оформление
    plt.xlabel('PC1', fontsize=10)
    plt.ylabel('PC2', fontsize=10)
    plt.title(title or f'PCA Visualization | Epoch {epoch}' if epoch else 'PCA Visualization', 
              fontsize=12, pad=15)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    return plt.gcf()


class Masker:
    def __init__(self, patch_size, mask_chance):
        self.patch_size = patch_size
        self.mask_chance = mask_chance
    
    def __call__(self, image):
        B, C, H, W = image.shape

        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size

        if pad_h != 0 or pad_w != 0:
            image = F.pad(image, (0, pad_w, 0, pad_h))

        _, _, H_pad, W_pad = image.shape

        h = H_pad // self.patch_size
        w = W_pad // self.patch_size

        mean_pixel = image.mean()

        mask = torch.rand(B, h, w, device=image.device)
        mask = torch.where(mask <= self.mask_chance, 0.0, 1.0)

        mask = mask.repeat_interleave(self.patch_size, dim=1)\
                    .repeat_interleave(self.patch_size, dim=2)  


        mask = mask.unsqueeze(1) 

        # masked = image * mask
        # masked = torch.where(masked == 0, mean_pixel, masked)

        noise = torch.randn_like(image) * image.std() + image.mean()

        masked = image.copy()
        masked[mask == 0] = noise[mask == 0]
        
        return masked, mask
    

def show_4_images(images, titles=None):
    """
    images: список из 4 изображений (numpy или torch.Tensor)
    titles: список из 4 заголовков (опционально)
    """
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    for i, ax in enumerate(axes.flat):
        img = images[i]

        # если это torch.Tensor → в numpy
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()

        # если CHW → HWC
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = img.transpose(1, 2, 0)

        ax.imshow(img.squeeze(), cmap="gray" if img.ndim == 2 or img.shape[-1] == 1 else None)
        ax.axis("off")

        if titles:
            ax.set_title(titles[i])

    plt.tight_layout()
    return fig


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, reduction = 'mean'):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, 1, H, W]
        # targets: [B, H, W]

        probs = torch.sigmoid(logits)
        targets = targets.unsqueeze(1).float()

        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    

def pr_auc_score(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = torch.sigmoid(pred_logits).detach().cpu().numpy()  # [B, 1, H, W]

    if targets.ndim == 4:
        targets = targets.squeeze(1)
    targets = targets.detach().cpu().numpy()  # [B, H, W]

    probs_flat = probs.reshape(-1)
    targets_flat = targets.reshape(-1)

    precision, recall, _ = precision_recall_curve(targets_flat, probs_flat)
    pr_auc = auc(recall, precision)
    return pr_auc


def PR_loss(pred_logits: torch.Tensor, targets: torch.Tensor) -> float:
    probs = F.sigmoid(pred_logits)
    smooth = 1e-6
    TP = (probs * targets).sum()
    precision = TP / (probs.sum() + smooth)
    recall = TP / (targets.sum() + smooth)
    pr_loss = 1 - (precision * recall)
    return pr_loss

class MAEWithDecoder(nn.Module):
    def __init__(self, mae_model, img_size=224, patch_size=16, in_chans=1):
        super().__init__()
        self.model = mae_model

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Linear(3072, 512)

    def unpatchify(self, x):
        B, N, D = x.shape
        p = self.patch_size
        h = w = int(N ** 0.5)  # 14

        x = x.view(B, h, w, p, p, self.in_chans)
        x = x.permute(0, 5, 1, 3, 2, 4)  # [B, C, h, p, w, p]
        x = x.reshape(B, self.in_chans, h * p, w * p)

        return x

    def forward(self, x):
        B = x.shape[0]

        feature_maps = []
        
        with torch.no_grad():
            x = self.model.patch_embed(x)  # [B, 196, 1024]

            for idx, block in enumerate(self.model.blocks):
                x = block(x)

                if idx in [7, 15, 23]:
                    fm = x.view(B, 14, 14, 1024).permute(0, 3, 1, 2)
                    feature_maps.append(fm)

            x = self.model.norm(x)

        x = torch.cat(feature_maps, dim=1)  # [B, 3072, 14, 14]
        x = x.flatten(2).transpose(1, 2)    # [B, 196, 3072]
        x = self.proj(x)                    # [B, 196, 512]

        for block in self.model.decoder_blocks:
            x = block(x)

        x = self.model.decoder_norm(x)
        x = self.model.decoder_pred(x)  # [B, 196, 256]

        x = self.unpatchify(x)  # [B, 1, 224, 224]

        return x

class AddBrChannel(A.ImageOnlyTransform):
    def __init__(self, scale, bias):
        super().__init__(True, 1.0)
        self.scale = scale
        self.b = bias
    def apply(self, image, **params):
        if image.ndim == 2:
            image = np.expand_dims(image, axis = 0)
            modified = image * self.scale + self.b
        else:
            modified = image[0] * self.scale + self.b
            modified = np.expand_dims(modified, axis = 0)

        modified /= modified.max()

        return np.concatenate([image, modified], axis = 0)
    

class MatchHistogram(A.ImageOnlyTransform):
    def __init__(self, ref_cdf, ref_bins):
        super().__init__(True, 1.0)
        self.cdf = ref_cdf
        self.bins = ref_bins

    def apply(self, image, **params):
        if image.ndim == 2:
            img = image.flatten()

            hist, bins = np.histogram(img, bins=1024, density=True)
            cdf = np.cumsum(hist)
            cdf = cdf / cdf[-1]

            interp_values = np.interp(cdf, self.cdf, self.bins[:-1])

            img_matched = np.interp(img, bins[:-1], interp_values)

            return img_matched.reshape(image.shape)
        else: 
            stack = []
            for img in image:
                img = img.flatten()

                hist, bins = np.histogram(img, bins=1024, density=True)
                cdf = np.cumsum(hist)
                cdf = cdf / cdf[-1]

                interp_values = np.interp(cdf, self.cdf, self.bins[:-1])

                img_matched = np.interp(img, bins[:-1], interp_values)

                stack.append(img_matched)

                img_matched = img_matched.reshape(image[0].shape)
            
            return np.stack(stack)
                


class CLAHEPrep(A.ImageOnlyTransform):
    def __init__(self):
        super().__init__(True, 1.0)
        self.clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(32,32)
            )

    def apply(self, image, **params):
        if image.max() < 250:
            image *= 255 / image.max()

        denoised = cv2.fastNlMeansDenoising(
                image.astype(np.uint8),
                None,
                h=8
            )
        
        augmented = self.clahe.apply(denoised) 

        return augmented / augmented.max()


def extract_ampl_phase(fft_im):
    # fft_im: complex tensor [B,C,H,W]
    amp = torch.abs(fft_im)
    pha = torch.angle(fft_im)
    return amp, pha


def low_freq_mutate(amp_src, amp_trg, L=0.1):
    """
    FDA low-frequency amplitude swap
    assumes amp tensors: [B,C,H,W]
    """
    _, _, h, w = amp_src.shape

    # shift DC to center
    amp_src = torch.fft.fftshift(amp_src, dim=(-2,-1))
    amp_trg = torch.fft.fftshift(amp_trg, dim=(-2,-1))

    b = int(np.floor(min(h,w)*L))
    c_h = h // 2
    c_w = w // 2

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    amp_src[..., h1:h2, w1:w2] = amp_trg[..., h1:h2, w1:w2]

    # shift back
    amp_src = torch.fft.ifftshift(amp_src, dim=(-2,-1))

    return amp_src


def FDA_source_to_target(src_img, trg_img, L=0.1):
    """
    src_img: [B,C,H,W]
    trg_img: [B,C,H,W]
    """

    # modern fft
    fft_src = torch.fft.fft2(src_img)
    fft_trg = torch.fft.fft2(trg_img)

    amp_src, pha_src = extract_ampl_phase(fft_src)
    amp_trg, _ = extract_ampl_phase(fft_trg)

    # swap low-freq amplitudes
    amp_src_mut = low_freq_mutate(
        amp_src.clone(),
        amp_trg.clone(),
        L=L
    )

    # recomposition
    fft_src_mut = amp_src_mut * torch.exp(1j * pha_src)

    # inverse fft
    src_in_trg = torch.fft.ifft2(fft_src_mut).real

    return src_in_trg



import random
import albumentations as A
from albumentations.core.transforms_interface import BasicTransform
import cv2

class RandomSizeCrop(BasicTransform):
    """
    Вырезает случайный кроп случайного размера в заданных рамках.
    
    Args:
        min_height (int): Минимальная высота кропа.
        max_height (int): Максимальная высота кропа.
        min_width (int): Минимальная ширина кропа.
        max_width (int): Максимальная ширина кропа.
        resize_to_original (bool): Если True, ресайзит кроп обратно к исходному размеру изображения.
                                   Если False, возвращает изображение измененного размера (может сломать батчинг).
        p (float): Вероятность применения.
    """
    def __init__(self, min_height, max_height, min_width, max_width, resize_to_original=True, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width
        self.resize_to_original = resize_to_original

    def apply(self, img, crop_height, crop_width, h_start, w_start, **params):
        # 1. Вычисляем координаты кропа
        height, width = img.shape[:2]
        
        # Ограничиваем размеры кропа, чтобы они не превышали изображение
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        # Случайная позиция начала кропа
        if height == crop_height:
            y1 = 0
        else:
            y1 = random.randint(0, height - crop_height)
            
        if width == crop_width:
            x1 = 0
        else:
            x1 = random.randint(0, width - crop_width)
            
        y2 = y1 + crop_height
        x2 = x1 + crop_width
        
        # 2. Делаем кроп
        cropped_img = img[y1:y2, x1:x2]
        
        # 3. Если нужно, ресайзим обратно к исходному размеру
        if self.resize_to_original:
            cropped_img = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_LINEAR)
            
        return cropped_img

    def apply_to_mask(self, mask, crop_height, crop_width, h_start, w_start, **params):
        # Та же логика для маски, но INTER_NEAREST для сохранения классов
        height, width = mask.shape[:2]
        crop_height = min(crop_height, height)
        crop_width = min(crop_width, width)
        
        if height == crop_height:
            y1 = 0
        else:
            y1 = random.randint(0, height - crop_height)
        if width == crop_width:
            x1 = 0
        else:
            x1 = random.randint(0, width - crop_width)
            
        y2 = y1 + crop_height
        x2 = x1 + crop_width
        
        cropped_mask = mask[y1:y2, x1:x2]
        
        if self.resize_to_original:
            cropped_mask = cv2.resize(cropped_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
        return cropped_mask

    def get_params_dependent_on_targets(self, params):
        # Генерируем случайные размеры один раз на вызов
        crop_height = random.randint(self.min_height, self.max_height)
        crop_width = random.randint(self.min_width, self.max_width)
        return {
            "crop_height": crop_height,
            "crop_width": crop_width,
            "h_start": 0, # не используется в логике выше, но требуется интерфейсом
            "w_start": 0
        }

    @property
    def targets_as_params(self):
        return ["image"]
    


class PercentileNormalize(A.ImageOnlyTransform):
    def __init__(self, p_low=2.5, p_high=97.5, always_apply=True, p=1.0):
        super().__init__(p=p)
        self.p_low = p_low
        self.p_high = p_high

    def apply(self, img, **params):
        img = img.astype(np.float32)

        low, high = np.percentile(img, [self.p_low, self.p_high])

        img = np.clip(img, low, high)

        img = (img - low) / (high - low + 1e-8)

        img = (img * 255.0).astype(np.uint8)

        return img
    

def compute_binary_iou(predictions,
                       targets,
                       threshold=0.5,
                       reduction="mean",
                       eps=1e-6):

    if predictions.dim() == 4:
        if predictions.shape[1] == 1:
            preds = (torch.sigmoid(predictions) > threshold)
            preds = preds.squeeze(1)
        else:
            raise ValueError(
                "Binary IoU expects 1 channel predictions"
            )
    else:
        preds = predictions.bool()

    if targets.dim() == 4:
        targets = targets.squeeze(1)

    targets = targets.bool()

    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    inter = (preds & targets).sum(dim=1).float()
    union = (preds | targets).sum(dim=1).float()

    iou = inter / (union + eps)

    empty_mask = union == 0
    iou[empty_mask] = 1.0

    if reduction == "none":
        return iou

    elif reduction == "mean":
        return iou.mean()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

def pr_auc_score(pred_logits: torch.Tensor,
                 targets: torch.Tensor,
                 reduction: str = "mean"):

    probs = torch.sigmoid(pred_logits).detach().cpu()

    if targets.ndim == 4:
        targets = targets.squeeze(1)

    targets = targets.detach().cpu()

    B = probs.shape[0]

    probs_flat = probs.reshape(B, -1).numpy()
    targets_flat = targets.reshape(B, -1).numpy()

    scores = []

    for b in range(B):
        precision, recall, _ = precision_recall_curve(
            targets_flat[b],
            probs_flat[b]
        )

        pr_auc = auc(recall, precision)
        scores.append(pr_auc)

    scores = torch.tensor(scores)

    if reduction == "none":
        return scores

    elif reduction == "mean":
        return scores.mean().item()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    


def metrix(pred_logits: torch.Tensor,
           targets: torch.Tensor,
           threshold: float = 0.5,
           reduction: str = "mean",
           eps: float = 1e-7):
    """
    Binary segmentation metrics.

    Args:
        pred_logits: [B, 1, H, W]
        targets:     [B, H, W] or [B, 1, H, W]
        threshold:   sigmoid threshold
        reduction:
            - "mean" -> scalar metrics
            - "none" -> metrics per image [B]

    Returns:
        dict with:
            - f1
            - precision
            - recall
    """

    probs = torch.sigmoid(pred_logits)

    preds = (probs > threshold).float()

    if targets.ndim == 3:
        targets = targets.unsqueeze(1)

    targets = targets.float()

    # flatten per-image
    preds = preds.reshape(preds.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)

    if reduction == "none":
        return f1, precision, recall
    elif reduction == "mean":
        return f1.mean().item(), precision.mean().item(), recall.mean().item()

    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    

class CoreDataset(torch.utils.data.Dataset):
    def __init__(self, path, train_ex : np.uint, val_ex : np.uint, val_size = 0.15, train_shuffle = True, train_transform = None, val_transform = None):
        self.path = path

        self.dataset_list = os.listdir(path)

        self.train_img_lists = {}
        self.train_msk_lists = {}

        self.val_img_lists = {}
        self.val_msk_lists = {}

        for dataset in self.dataset_list:
            img_list = sorted(os.listdir(self.path + "/" + dataset + "/images"))
            msk_list = sorted(os.listdir(self.path + "/" + dataset + "/masks"))

            assert len(img_list) == len(msk_list)

            p = int(len(img_list) * val_size)

            indices = list(range(len(img_list)))
            random.shuffle(indices)

            val_indices = indices[:p]
            train_indices = indices[p:]

            self.val_img_lists[dataset] = [img_list[i] for i in val_indices]
            self.val_msk_lists[dataset] = [msk_list[i] for i in val_indices]

            self.train_img_lists[dataset] = [img_list[i] for i in train_indices]
            self.train_msk_lists[dataset] = [msk_list[i] for i in train_indices]

        self.train_ex = train_ex
        self.val_ex = val_ex
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batchsize = len(self.dataset_list)
        self.train_shuffle = train_shuffle
        
        self.tr = True
        
    def train(self):
        self.tr = True
    
    def eval(self):
        self.tr = False

    def __len__(self):
        if self.tr:
            return self.train_ex
        else:
            return self.val_ex

    def __iter__(self):
        if self.tr:
            for ex in range(self.train_ex):
                imgs = []
                msks = []
                flgs = []
                for dataset in self.dataset_list:
                    img_list = self.train_img_lists[dataset]
                    msk_list = self.train_msk_lists[dataset]

                    idx = random.randint(0, len(img_list) - 1)
                    img_path = self.path + "/" + dataset + "/images/" + img_list[idx]
                    msk_path = self.path + "/" + dataset + "/masks/" + msk_list[idx]

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                    
                    if self.train_transform:
                        transformed = self.train_transform(image=img, target=msk)
                        img = transformed['image']
                        msk = transformed['target']
                    
                    imgs.append(torch.from_numpy(img))
                    msks.append(torch.from_numpy(msk))
                    flgs.append(dataset)

                yield {
                    "image": torch.stack(imgs),  
                    "target": torch.stack(msks),
                    "flag": flgs,       
                    }
        else:
            for ex in range(self.val_ex):
                imgs = []
                msks = []
                flgs = []
                for dataset in self.dataset_list:
                    img_list = self.val_img_lists[dataset]
                    msk_list = self.val_msk_lists[dataset]

                    idx = random.randint(0, len(img_list) - 1)
                    img_path = self.path + "/" + dataset + "/images/" + img_list[idx]
                    msk_path = self.path + "/" + dataset + "/masks/" + msk_list[idx]

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                    
                    if self.val_transform:
                        transformed = self.val_transform(image=img, target=msk)
                        img = transformed['image']
                        msk = transformed['target']
                    
                    imgs.append(torch.from_numpy(img))
                    msks.append(torch.from_numpy(msk))
                    flgs.append(dataset)
                
                if self.tr and self.train_shuffle:
                    combined = list(zip(imgs, msks, flgs))
                    random.shuffle(combined)
                    imgs, msks, flgs = map(list, zip(*combined))

                yield {
                    "image": torch.stack(imgs),  
                    "target": torch.stack(msks),
                    "flag": flgs,       
                    }
                



class ThreadedLoader:
    def __init__(self, dataset, max_queue_size=10):

        self.dataset = dataset
        self.queue = queue.Queue(maxsize=max_queue_size)

        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        for batch in self.dataset: 

            if self.stop_event.is_set():
                break

            self.queue.put(batch)

        self.queue.put(None)  

    def start(self):
        self.thread.start()
        return self

    def __iter__(self):
        return self

    def __next__(self):

        item = self.queue.get()

        if item is None:
            raise StopIteration

        return item

    def stop(self):
        self.stop_event.set()
    
    def __len__(self):
        if self.dataset.tr:
            return self.dataset.train_ex
        else:
            return self.dataset.val_ex