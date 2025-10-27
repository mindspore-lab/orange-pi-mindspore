# eval.py
# SegFormer (MindNLP) on ADE20K: evaluation (mIoU/PA) on validation, or inference on release_test
import os
import argparse
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import mindspore as ms
from mindspore import Tensor, ops, context

from mindnlp.transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# -------------------
# Default Config
# -------------------
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
NUM_CLASSES = 150
IGNORE_INDEX = 255

# Kaggle 解压默认根目录（按你的环境修改或用命令行参数覆盖）
DEFAULT_KAGGLE_ROOT = "/home/ma-user/.cache/kagglehub/datasets/ipythonx/ade20k-scene-parsing/versions/2"

def add_args():
    p = argparse.ArgumentParser(description="SegFormer on ADE20K (MindSpore/MindNLP)")
    p.add_argument("--mode", choices=["eval", "infer"], default="eval",
                   help="eval: 在 validation 上评测; infer: 对 release_test/testing 做推理（无指标）")
    p.add_argument("--kaggle_root", type=str, default=DEFAULT_KAGGLE_ROOT,
                   help="Kaggle 数据解压根目录（包含 ADEChallengeData2016/ 与 release_test/）")
    p.add_argument("--device", type=str, default="Ascend",
                   help="设备: Ascend/GPU/CPU")
    p.add_argument("--model", type=str, default=MODEL_NAME,
                   help="Hugging Face 模型权重名称")
    p.add_argument("--out_dir", type=str, default="/home/ma-user/work/segformer_out",
                   help="仅在 --mode infer 时使用：输出掩码与可视化目录")
    return p.parse_args()

# -------------------
# IO helpers
# -------------------
def list_pairs(img_dir: str, ann_dir: str) -> List[Tuple[str, str]]:
    """按同名匹配 validation 的 (image, annotation) 对。"""
    img_paths = sorted(glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True))
    pairs = []
    for ip in img_paths:
        stem = Path(ip).stem
        ap = os.path.join(ann_dir, stem + ".png")
        if os.path.isfile(ap):
            pairs.append((ip, ap))
    return pairs

def list_images(img_dir: str) -> List[str]:
    """列出测试集所有图片路径。"""
    return sorted(glob(os.path.join(img_dir, "**", "*.jpg"), recursive=True))

# -------------------
# Metrics
# -------------------
def fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]
    hist = np.bincount(
        num_classes * target.astype(int) + pred.astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist

def compute_miou_pa(hist: np.ndarray) -> Tuple[float, float]:
    with np.errstate(divide="ignore", invalid="ignore"):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.nanmean(iu)
        pa = np.diag(hist).sum() / hist.sum()
    return float(miou), float(pa)

# -------------------
# Visualization
# -------------------
def palette_150() -> np.ndarray:
    rng = np.random.default_rng(0)
    pal = rng.integers(0, 256, size=(NUM_CLASSES, 3), dtype=np.uint8)
    pal[0] = np.array([0, 0, 0], dtype=np.uint8)  # 背景固定为黑色
    return pal

def save_pred_mask(pred_np: np.ndarray, save_mask_path: str, save_vis_path: str = None):
    Image.fromarray(pred_np.astype(np.uint8), mode="L").save(save_mask_path)
    if save_vis_path:
        pal = palette_150()
        color = pal[pred_np.clip(0, NUM_CLASSES - 1)]
        Image.fromarray(color, mode="RGB").save(save_vis_path)

# -------------------
# Model inference helper
# -------------------
def logits_to_pred_np(logits: Tensor) -> np.ndarray:
    """logits [1, C, h, w] -> pred_np [h, w] uint8, 兼容不同 MindSpore 版本的 argmax。"""
    try:
        pred = ops.argmax(logits, 1)    # 某些版本只支持位置参数
    except TypeError:
        pred = ops.Argmax(axis=1)(logits)
    pred_np = pred.asnumpy().astype(np.uint8)
    if pred_np.ndim == 3:
        pred_np = pred_np[0]            # 去掉 batch 维
    return pred_np

# -------------------
# Main
# -------------------
def main():
    args = add_args()

    # 设置设备
    # 新接口建议：ms.set_device；为兼容旧版本也保留 set_context
    try:
        ms.set_device(args.device)
    except Exception:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    print(f"[Info] Device: {args.device}")

    # 模型与处理器
    print("[Info] Loading model & processor ...")
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model)
    model.set_train(False)

    # 路径
    ade_img_val = os.path.join(args.kaggle_root, "ADEChallengeData2016/images/validation")
    ade_ann_val = os.path.join(args.kaggle_root, "ADEChallengeData2016/annotations/validation")
    test_img_dir = os.path.join(args.kaggle_root, "release_test/testing")

    if args.mode == "eval":
        # -------- Evaluation on validation --------
        pairs = list_pairs(ade_img_val, ade_ann_val)
        if not pairs:
            raise RuntimeError(
                f"未找到 validation 样本，请检查路径：\n"
                f"  images: {ade_img_val}\n  annots: {ade_ann_val}"
            )
        print(f"[Info] Eval samples: {len(pairs)}")

        hist = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

        for i, (ip, ap) in enumerate(pairs, 1):
            # 读图
            img = Image.open(ip).convert("RGB")
            lab = Image.open(ap)  # 通常为单通道索引图

            # 预处理 + 前向
            inputs = processor(img, return_tensors="ms")
            outputs = model(inputs["pixel_values"])
            pred_np = logits_to_pred_np(outputs.logits)  # [h_out, w_out]

            # 将 GT 最近邻下采样到 logits 尺寸，避免插值引入新类别
            if lab.mode not in ("L", "P"):
                lab = lab.convert("L")
            h_out, w_out = int(pred_np.shape[-2]), int(pred_np.shape[-1])
            lab_ds = lab.resize((w_out, h_out), resample=Image.NEAREST)
            lab_np = np.array(lab_ds, dtype=np.uint8)
            lab_np = np.where(lab_np < NUM_CLASSES, lab_np, IGNORE_INDEX)
            
            do_reduce = getattr(processor, "do_reduce_labels", None)
            if do_reduce is None:
                do_reduce = getattr(processor, "reduce_labels", False)
            if do_reduce:
                lab_np = np.where(lab_np == 0, 255, lab_np)
                lab_np = np.where(lab_np != 255, lab_np - 1, 255)
                
            # 累积混淆矩阵
            hist += fast_hist(pred_np, lab_np, NUM_CLASSES, IGNORE_INDEX)

            if i % 100 == 0 or i == len(pairs):
                miou, pa = compute_miou_pa(hist)
                print(f"[{i}/{len(pairs)}] mIoU={miou:.4f}, PA={pa:.4f}")

        miou, pa = compute_miou_pa(hist)
        print("========== Final (validation) ==========")
        print(f"mIoU: {miou:.4f}")
        print(f"Pixel Acc: {pa:.4f}")

    elif args.mode == "infer":
        # -------- Inference on release_test/testing --------
        imgs = list_images(test_img_dir)
        if not imgs:
            raise RuntimeError(
                f"未在测试集目录找到图片：{test_img_dir}\n"
                f"注：release_test/testing 不含标注，仅可做推理不可评测。"
            )
        out_mask = os.path.join(args.out_dir, "masks")
        out_vis = os.path.join(args.out_dir, "vis")
        os.makedirs(out_mask, exist_ok=True)
        os.makedirs(out_vis, exist_ok=True)

        print(f"[Info] Infer images: {len(imgs)}  ->  out: {args.out_dir}")
        for i, ip in enumerate(imgs, 1):
            img = Image.open(ip).convert("RGB")
            inputs = processor(img, return_tensors="ms")
            outputs = model(inputs["pixel_values"])
            pred_np = logits_to_pred_np(outputs.logits)

            stem = Path(ip).stem
            save_pred_mask(
                pred_np,
                os.path.join(out_mask, f"{stem}.png"),
                os.path.join(out_vis,  f"{stem}.jpg"),
            )
            if i % 50 == 0 or i == len(imgs):
                print(f"  saved {i}/{len(imgs)}")

        print("Done. Results in:", args.out_dir)

    else:
        raise ValueError("args.mode 必须是 'eval' 或 'infer'。")

if __name__ == "__main__":
    main()
