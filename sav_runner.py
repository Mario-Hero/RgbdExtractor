# -*- coding: utf-8 -*-
"""
Stereo Any Video 推理封装 (PyTorch + CUDA)

- 模型代码位于 third_party/stereoanyvideo, 依赖 VDA 权重已放在其 checkpoints 下
- 本文件只暴露两个能力: 单帧/滑窗推理, 以及 hequirect 球面视差->距离换算
- 滑窗: 每 20 帧一组, 步进 10 帧, 每组只保留中间部分, 兼顾时间一致性与显存
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "third_party" / "stereoanyvideo"
DEFAULT_CKPT = ROOT / "weights" / "StereoAnyVideo_MIX.pth"
DEFAULT_VDA = REPO / "models" / "Video-Depth-Anything" / "checkpoints" / "video_depth_anything_vits.pth"

if str(ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(ROOT / "third_party"))


def round32(v: float) -> int:
    return max(32, int(round(v / 32.0)) * 32)


class SAVModel:
    def __init__(self, ckpt: str = "", vda: str = "", device: str = "cuda", amp: bool = True):
        import torch

        self.torch = torch
        self.device = device
        self.amp = amp
        ckpt = str(ckpt or DEFAULT_CKPT)
        vda = str(vda or DEFAULT_VDA)
        if not Path(ckpt).is_file():
            raise FileNotFoundError(
                f"缺少主权重: {ckpt}\n请从 Google Drive 下载 StereoAnyVideo_MIX.pth 后放到 weights/")
        if not Path(vda).is_file():
            raise FileNotFoundError(f"缺少 VDA 权重: {vda}")

        # DepthExtractor 用相对路径加载 VDA 权重, 构造时临时切到仓库根目录
        old = os.getcwd()
        os.chdir(str(REPO))
        try:
            from stereoanyvideo.models.core.stereoanyvideo import StereoAnyVideo
            model = StereoAnyVideo(mixed_precision=amp)
        finally:
            os.chdir(old)

        try:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
        if "model" in state:
            state = state["model"]
        if "state_dict" in state:
            state = state["state_dict"]
            state = {"module." + k: v for k, v in state.items()}
        first = next(iter(state))
        if first.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f"[sav] loaded {Path(ckpt).name}  amp={amp}")

        self.model = model.to(device).eval()
        if torch.cuda.is_available() and device.startswith("cuda"):
            print(f"[sav] GPU {torch.cuda.get_device_name(0)}")

    def _prep(self, lefts, rights, in_w: int, in_h: int):
        torch = self.torch
        ls, rs = [], []
        for l, r in zip(lefts, rights):
            ls.append(cv2.resize(l, (in_w, in_h), interpolation=cv2.INTER_AREA))
            rs.append(cv2.resize(r, (in_w, in_h), interpolation=cv2.INTER_AREA))
        l = torch.from_numpy(np.stack(ls)).permute(0, 3, 1, 2).float()   # BGR 顺序无影响: 三通道同时送
        r = torch.from_numpy(np.stack(rs)).permute(0, 3, 1, 2).float()
        # BGR -> RGB
        l = l[:, [2, 1, 0]].contiguous().to(self.device)
        r = r[:, [2, 1, 0]].contiguous().to(self.device)
        return l, r

    def infer_window(self, lefts, rights, in_w: int, in_h: int, iters: int = 12):
        """lefts/rights: 等长 BGR uint8 列表 -> (T, in_h, in_w) float32 视差 (px, 原始输入分辨率)"""
        torch = self.torch
        l, r = self._prep(lefts, rights, in_w, in_h)
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                out = self.model.forward(l[None], r[None], iters=iters, test_mode=True)
        disp = out[:, 0, 0].float().cpu().numpy()
        return np.clip(disp, 0.0, None)

    def infer_single(self, left, right, in_w: int, in_h: int, iters: int = 12, ctx: int = 8):
        """单帧: 模型依赖时间上下文, T=1 结果不稳定, 故把同一对帧重复 ctx 次后取中位视差"""
        reps = max(2, int(ctx))
        disp = self.infer_window([left] * reps, [right] * reps, in_w, in_h, iters)
        return np.median(disp, axis=0)


def window_plan(n: int, win: int = 20, overlap: int = 5):
    """生成 (start, end, keep_end) 列表; start..end 送入模型, start..keep_end 输出。
    每组保留右侧 overlap 帧作为下一组的上下文, 末组保留到结尾, 保证覆盖全部帧。"""
    win = max(4, win)
    overlap = max(1, min(overlap, win // 4))
    plan = []
    emitted = 0
    while emitted < n:
        s = max(0, emitted - overlap)
        e = min(s + win, n)
        keep_end = e if e == n else max(emitted + 1, e - overlap)
        plan.append((s, e, keep_end))
        emitted = keep_end
    return plan


def disp_to_range_hequirect(disp_full: np.ndarray, baseline: float,
                            min_disp_px: float = 0.3, min_range: float = 0.05,
                            max_range: float = 50.0, min_cos: float = 0.0,
                            clamp: bool = False) -> np.ndarray:
    """hequirect 视差(原始分辨率像素) -> 相机中心距离 (米)
    水平像素差 = 方位角差 δ = disp·π/W, 按纬度行独立三角化:
      r = B · cos(θ_R) / sin(δ),  θ_R = θ - δ
    (曾改为同纬双视线 3D 三角化 r=B·num/|u_L×u_R|², 但 equirect 双目同行匹配
    的两条视线在空间并不相交, 分子引入 -sin²φ·sinθ 项: 右侧 (θ>0, |φ| 大)
    深度被系统性算小甚至变负, 形成右侧 U 形近值黑圈, 故回退到行独立公式)
    边缘 θ_R→±90° 视线与基线共线, cos(θ_R)<min_cos 判无效, 交给空洞填充

    默认返回**原始米制距离**: 不做 near/far 钳制, 0 表示无效
    (钳制放在输出/编码阶段, 这样缓存的原始毫米可反复换 near/far 复用);
    clamp=True 时才把 <=min_range 的贴脸点判无效、>max_range 钳到 max_range
    """
    H, W = disp_full.shape
    xs = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    theta = (xs / W - 0.5) * np.pi          # 经度: -90°..+90°
    delta = disp_full.astype(np.float64) * (np.pi / W)
    theta_R = theta - delta
    sin_d = np.sin(delta)
    cos_r = np.cos(theta_R)

    valid = (disp_full > min_disp_px) & (sin_d > 1e-9)
    if min_cos > 0:
        # 视线接近与基线共线 (±90° 附近) 三角化病态, 判无效交给空洞填充
        valid &= cos_r > min_cos
    r = np.zeros_like(delta)
    r[valid] = baseline * cos_r[valid] / sin_d[valid]
    if clamp:
        r[r <= min_range] = min_range
        r[r > max_range] = max_range
    return r.astype(np.float32)
