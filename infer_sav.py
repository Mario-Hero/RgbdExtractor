# -*- coding: utf-8 -*-
"""
VR180 (hequirect) 双目单帧 -> 深度图 + 点云
模型: Stereo Any Video (PyTorch + CUDA, 时间一致视频立体匹配)

流程:
  1) 整幅 hequirect 左右图等比例缩放到 --size 宽度 (高度按 2:1 自动推算), 单帧送入模型
  2) 视差按宽度比例缩放回原始分辨率
  3) 球面几何视差->距离: hequirect 水平像素差即方位角差 delta_theta,
     r = B * cos(theta_R) / sin(delta_theta)
  4) 导出: 16bit 深度 PNG (毫米) + 彩色预览 + PLY 点云

用法:
  python infer_sav.py --left frames/makeup/L/L_00000.png --right frames/makeup/R/R_00000.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from sav_runner import SAVModel, round32, disp_to_range_hequirect


def save_depth16(depth: np.ndarray, path: Path):
    cv2.imwrite(str(path), np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))


def save_depth_preview(depth: np.ndarray, path: Path):
    valid = depth > 0
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if np.any(valid):
        lo, hi = np.percentile(depth[valid], 1), np.percentile(depth[valid], 99)
        v = np.clip((depth - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
        v[~valid] = 0
        out = cv2.applyColorMap(v, cv2.COLORMAP_TURBO)
        out[~valid] = 0
    cv2.imwrite(str(path), out)


def save_disp_preview(disp: np.ndarray, path: Path):
    valid = disp > 0
    vis = np.zeros_like(disp, dtype=np.uint8)
    if np.any(valid):
        hi = np.percentile(disp[valid], 99)
        vis = np.clip(disp / max(hi, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO))


def save_ply(rgb_bgr_heq: np.ndarray, depth_heq: np.ndarray, path: Path, step: int = 4):
    """hequirect RGB + 距离 -> binary PLY (球面反投影)"""
    H, W = depth_heq.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    theta = (xs / W - 0.5) * np.pi
    phi = (0.5 - ys / H) * np.pi
    r = depth_heq[ys, xs]
    valid = r > 0
    theta, phi, r = theta[valid], phi[valid], r[valid].astype(np.float64)
    X = r * np.cos(phi) * np.sin(theta)
    Y = r * np.sin(phi)
    Z = r * np.cos(phi) * np.cos(theta)
    c = rgb_bgr_heq[ys[valid], xs[valid]]
    B, G, R = c[:, 0], c[:, 1], c[:, 2]

    n = len(r)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with open(path, "wb") as f:
        f.write(header)
        data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                  ("r", "u1"), ("g", "u1"), ("b", "u1")])
        data["x"], data["y"], data["z"] = X, Y, Z
        data["r"], data["g"], data["b"] = R, G, B
        f.write(data.tobytes())
    print(f"[ply] {path}  {n} 点")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default=r"frames\makeup\L\L_00000.png")
    ap.add_argument("--right", default=r"frames\makeup\R\R_00000.png")
    ap.add_argument("--model", default="", help="默认 weights/StereoAnyVideo_MIX.pth")
    ap.add_argument("--vda", default="", help="Video Depth Anything 权重路径")
    ap.add_argument("--size", type=int, default=640,
                    help="推理宽度 (高度按原图比例推算, 32 对齐); 0=原分辨率直送(大图易爆显存)")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--ctx", type=int, default=8,
                    help="单帧时重复送入的帧数 (模型依赖时间上下文, T=1 不稳定)")
    ap.add_argument("--no-amp", action="store_true", help="关闭 BF16 自动混合精度")
    ap.add_argument("--baseline", type=float, default=0.065, help="双目基线 (米)")
    ap.add_argument("--min-range", type=float, default=0.05)
    ap.add_argument("--max-range", type=float, default=10.0)
    ap.add_argument("--min-cos", type=float, default=0.1,
                    help="几何可靠性阈值: cos(theta_R) 太小判为无效; 0=关闭")
    ap.add_argument("--ply-step", type=int, default=4, help="点云抽稀步长 (px)")
    ap.add_argument("--no-ply", action="store_true", help="跳过点云导出")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    left = cv2.imread(args.left, cv2.IMREAD_COLOR)
    right = cv2.imread(args.right, cv2.IMREAD_COLOR)
    if left is None or right is None:
        sys.exit("读取左右图失败")
    H, W = left.shape[:2]
    print(f"[input] {args.left}  {W}x{H}")

    in_w = round32(args.size) if args.size else W
    in_h = round32(in_w * H / W) if args.size else H

    model = SAVModel(args.model, args.vda, amp=not args.no_amp)
    disp_in = model.infer_single(left, right, in_w, in_h, args.iters, ctx=args.ctx)

    if (in_w, in_h) == (W, H):
        disp_full = disp_in
    else:
        disp_full = cv2.resize(disp_in, (W, H), interpolation=cv2.INTER_LINEAR) * (W / in_w)

    depth = disp_to_range_hequirect(disp_full, args.baseline,
                                    min_range=args.min_range, max_range=args.max_range,
                                    min_cos=args.min_cos, clamp=True)
    valid = depth > 0
    if np.any(valid):
        print(f"[depth] 有效像素 {valid.mean()*100:.1f}%  "
              f"范围 [{depth[valid].min():.2f}, {np.median(depth[valid]):.2f}, "
              f"{np.percentile(depth[valid], 99):.2f}] m")
    else:
        print("[depth] 全无效")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.left).stem
    cv2.imwrite(str(out / f"{stem}_rgb.png"), left)
    save_depth16(depth, out / f"{stem}_depth16.png")
    save_depth_preview(depth, out / f"{stem}_depth_preview.png")
    save_disp_preview(disp_full, out / f"{stem}_disp_preview.png")
    np.save(out / f"{stem}_depth.npy", depth)
    if not args.no_ply:
        save_ply(left, depth, out / f"{stem}.ply", step=args.ply_step)
    print(f"[done] 输出目录: {out.resolve()}")


if __name__ == "__main__":
    main()
