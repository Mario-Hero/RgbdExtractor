# -*- coding: utf-8 -*-
r"""
VR180 双目帧序列 -> 深度 SBS 视频 (H.265 8/10/12bit), 模型 = Stereo Any Video

输出布局 (分辨率与原视频一致, 左右各半):
  | 左视图原图 (RGB) | 深度图 (灰度: 深度对数量化到 0..max_code, --depth-encode linear 可切回线性) |

gray10 算法 (默认 log 对数量化, 相对误差恒定, 近处精度高):
  深度 d 米 -> v = log(d/near) / log(far/near), 无效像素(d<=0)强制 v=1 (最远)
  码值 c = round(v * max_code), max_code = 255/1023/4095 (--depth-bits 8/10/12)
  gray10mp4 打包: 码值平面以 gray/gray10le/gray12le 直通 ffmpeg (无 rgb48 转换失真), 无损 HEVC
  gray10/mkv 打包: rgb48le 每通道 uint16 = c << (16-bits), 右侧整幅亮度(Y)即深度码值
  播放端从亮度读出 c, 深度 = near * (far/near)^(c/max_code)
  (--depth-encode linear 时: v = (d-near)/(far-near), 反解 d = near + (far-near)*c/max_code)

用法:
  python process_video.py --input-video "C:\path\to\makeup.mp4" --start 0 --end 5 ^
      --out out\gray10\makeup_5s_gray10.mp4 --max-range 5
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from sav_runner import SAVModel, round32, window_plan, disp_to_range_hequirect

MIN_DISP_PX = 0.05
GRAY10_MAX = 1023

# Windows: ffmpeg 子进程隐藏控制台窗口, 但保留 stdin/stdout/stderr 继承
# (用 STARTF_USESHOWWINDOW+SW_HIDE 而非 CREATE_NO_WINDOW, 否则 CLI 下编码器报错信息看不到)
if os.name == "nt":
    NO_WINDOW = subprocess.STARTUPINFO()
    NO_WINDOW.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    NO_WINDOW.wShowWindow = subprocess.SW_HIDE
else:
    NO_WINDOW = None


def gray_in_pix(bits: int) -> str:
    """rawvideo 输入像素格式: 8bit 用 gray, 10/12bit 用 grayNle"""
    return "gray" if bits <= 8 else f"gray{bits}le"


def yuv_out_pix(bits: int) -> str:
    """HEVC 输出像素格式: 8bit 用 yuv420p, 10/12bit 用 yuv420pNle"""
    return "yuv420p" if bits <= 8 else f"yuv420p{bits}le"


def time_range_fields(start: float, end: float) -> dict:
    """播放区间 (秒). start=0 且 end 未指定(到末尾) 时不写, 播放端保持整段循环.

    end_time=0 表示播放到颜色视频末尾.
    """
    if start == 0 and end < 0:
        return {}
    return {"start_time": round(float(start), 6),
            "end_time": round(float(end), 6) if end > 0 else 0}


def nearest_fill(img: np.ndarray, valid: np.ndarray, max_dist: int = 0) -> np.ndarray:
    """把无效像素填成最近有效像素的值 (距离变换, 精确最近邻)"""
    if valid.all() or not valid.any():
        return img
    invalid = np.logical_not(valid)
    dist, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    ys, xs = np.nonzero(valid)
    sel = invalid if max_dist <= 0 else (invalid & (dist <= float(max_dist)))
    if not sel.any():
        return img
    out = img.copy()
    lab = labels[sel] - 1
    out[sel] = img[ys[lab], xs[lab]]
    return out


def encode_depth_gray10(depth: np.ndarray, near: float, far: float, out_w: int,
                        encode: str = "log", bits: int = 10) -> np.ndarray:
    """N bit 码值平面 (gray10le/gray12le 直通, 值 = 码值 0..max_code)
    depth: H x out_w 米制深度, 返回 H x out_w uint16

    直通 gray 输入绕开 swscale rgb48->yuv420pNle 转换的 +0.4%/级 相对增益失真
    (实测 4095/4080 满量程失配, 顶部饱和; gray 直通逐码 bit-exact)。
    rgb48 打包路径 (gray10 SBS/mkv) 仍由调用方自行做 code << (16-bits) 展开。

    encode="linear": v = (d-near)/(far-near)          绝对误差恒定 (far-near)/max_code
    encode="log":    v = log(d/near)/log(far/near)    相对误差恒定, 近处精度高 (推荐)
                     反解 d = near * (far/near)^v; 0.5m 处误差 2.2mm vs 线性 4.8mm
    """
    max_code = (1 << bits) - 1
    if encode == "log":
        d = np.clip(depth.astype(np.float64), near, far)
        v = np.log(d / near) / np.log(far / near)
        v[depth <= 0] = 1.0                       # 无效深度按最远端处理, 避免半径 0 穿出相机
    else:
        v = np.clip((depth.astype(np.float64) - near) / (far - near), 0.0, 1.0)
        v[depth <= 0] = 1.0                       # 无效深度按最远端处理, 避免半径 0 穿出相机
    code = np.round(v * max_code).astype(np.uint16)
    assert code.shape[1] == out_w, (code.shape, out_w)
    return code


def resize_depth(depth: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """推理网格深度 -> 输出分辨率 (纯几何缩放)
    1) 缩小用 INTER_AREA 抗锯齿, 等大等价拷贝, 放大用双线性;
    2) 残留的无效深度(0)用最近有效值填满: 0 码值在播放端会还原成 near, 顶成凸向观众的尖刺;
    3) 兜底抬到全图最小有效深度, 杜绝 0。
    """
    in_w, in_h = depth.shape[1], depth.shape[0]
    interp = cv2.INTER_AREA if (out_w < in_w or out_h < in_h) else cv2.INTER_LINEAR
    d = cv2.resize(depth, (out_w, out_h), interpolation=interp).astype(np.float32)
    valid = d > 0
    if not valid.any():
        return d
    if not valid.all():
        d = nearest_fill(d, valid, 0)                    # 0 = 不限距离, 全部填满
    return np.maximum(d, float(d[d > 0].min())).astype(np.float32)


def probe_video(path) -> tuple[int, int, float, int]:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
                        "-of", "default=noprint_wrappers=1", str(path)],
                       capture_output=True, text=True, startupinfo=NO_WINDOW)
    info = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    if "width" not in info:
        sys.exit(f"ffprobe 读取失败: {r.stderr.strip()}")
    num, _, den = info.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1)
    return int(info["width"]), int(info["height"]), fps, int(info.get("nb_frames") or 0)


def iter_sbs_frames(path, fw: int, fh: int, i0: int, n: int, fps: float):
    lw = fw // 2
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-ss", f"{i0 / fps:.6f}", "-i", str(path),
           "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, startupinfo=NO_WINDOW)
    if p.stdout is None:
        sys.exit("ffmpeg 解码启动失败")
    fsize = fw * fh * 3
    try:
        for _ in range(n):
            buf = p.stdout.read(fsize)
            if len(buf) < fsize:
                break
            arr = np.frombuffer(buf, np.uint8).reshape(fh, fw, 3)
            yield arr[:, :lw], arr[:, lw:2 * lw]
    finally:
        p.stdout.close()
        p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default="", help="含 L/ 与 R/ 子目录")
    ap.add_argument("--input-video", default="", help="直接读 SBS 视频(左右各半), 替代 --frames-dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--model", default="", help="默认 weights/StereoAnyVideo_MIX.pth")
    ap.add_argument("--vda", default="", help="Video Depth Anything 权重路径")
    ap.add_argument("--depth-size", type=int, default=640,
                    help="推理宽度 (高度按原图比例推算, 32 对齐); 12GB 显存建议 640")
    ap.add_argument("--window", type=int, default=20, help="时间窗口帧数 (Stereo Any Video 推荐 20)")
    ap.add_argument("--iters", type=int, default=12, help="GRU 迭代次数")
    ap.add_argument("--no-amp", action="store_true", help="关闭 BF16 自动混合精度")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=-1.0)
    ap.add_argument("--baseline", type=float, default=0.065)
    ap.add_argument("--swap-eyes", action="store_true",
                    help="以右眼为基准推理: 送入模型前水平翻转并交换左右眼 (交换后视差为负会被模型截断, "
                         "镜像后视差重新为正), 视差翻回右眼图像坐标再换算深度, 结果=右眼距离; "
                         "只影响推理输入, 颜色仍取左半")
    ap.add_argument("--both-eyes", action="store_true",
                    help="仅 --pack gray10mp4: 左右眼深度都导出, 每个窗口推理两次 (正常 + 翻转交换), "
                         "深度视频左右拼接 (左半=左眼, 右半=右眼, 宽 2*depth-size); "
                         "json 记 depth_layout=sbs, 颜色直接用原版 SBS, 播放端双球壳分眼渲染")
    ap.add_argument("--min-range", type=float, default=0.05)
    ap.add_argument("--max-range", type=float, default=5.0, help="深度截断(米);")
    ap.add_argument("--crf", type=int, default=16)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--encoder", default="libx265", choices=["auto", "libx265", "hevc_nvenc"],
                    help="默认 libx265; hevc_nvenc 快得多但需与模型抢显存, 建议仅配合 --depth-from 使用")
    ap.add_argument("--pack", default="gray10",
                    choices=["gray10", "gray10mkv", "gray10mp4", "depthnpy"],
                    help="gray10=单轨 SBS mp4(左彩右深); gray10mkv=单 MKV 双视频轨; "
                         "gray10mp4=<out>_depth.mp4 深度 10bit 无损 + <out>.json; 颜色不导出, 直接用原版 SBS (--input-video), 播放端裁左半; "
                         "depthnpy=仅导出深度缓存 float32 原始毫米 .npy 到 --depth-cache (模型分辨率), 不生成视频")
    ap.add_argument("--depth-encode", default="log", choices=["linear", "log"],
                    help="深度码值量化: log=对数(相对误差恒定, 近处精度高, 默认); "
                         "linear=线性(绝对误差恒定, 旧版行为). 播放端按 meta json 的 encode 字段反解")
    ap.add_argument("--depth-bits", type=int, default=10, choices=[8, 10, 12],
                    help="深度码值位深: 8=255 级; 10=1023 级(默认); 12=4095 级(yuv420p12le 无损, "
                         "仅 libx265, 播放端按 ffprobe 像素格式自动识别)")
    ap.add_argument("--color-file", default="",
                    help="写入 json 的彩色视频路径; 留空=用 --input-video (原版 SBS). "
                         "可填 GUI 导出的单目 8bit 左半幅视频, 播放端不必解码 8K SBS, 更流畅")
    ap.add_argument("--color-layout", default="", choices=["", "sbs_left", "tb_top", "mono"],
                    help="json 的 color_layout: sbs_left=原版 SBS 播放端裁左半; "
                         "mono=已是单目全幅 (配合 --color-file); "
                         "留空=有 --color-file 时为 mono, 否则 sbs_left")
    ap.add_argument("--depth-cache", default="",
                    help="深度缓存目录 (float32 .npy, 原始毫米, 未钳制); "
                         "depthnpy 模式下该目录就是导出目录, 写同样格式的 .npy")
    ap.add_argument("--depth-from", default="",
                    help="从该目录读取 float32 .npy 深度缓存 (原始毫米), 跳过模型推理直接合成视频")
    ap.add_argument("--fill-max", type=int, default=16, help="视差域小空洞填充最大距离(px); 0=关闭")
    ap.add_argument("--fill-depth-dist", type=int, default=0,
                    help="深度域填充最大距离(px); 0=全部填满")
    ap.add_argument("--min-cos", type=float, default=0.1,
                    help="cos(theta_R) 可靠性阈值; 0=关闭")
    ap.add_argument("--edge-fix", action=argparse.BooleanOptionalAction, default=False,
                    help="边缘修正(默认关闭): 把左右两侧被模型估成 5~20cm 的过近深度"
                         "视为无效, 由内侧有效像素外推, 避免渲染成指向视点的尖刺; 角度见 --side-deg")
    ap.add_argument("--nadir-deg", type=float, default=80.0,
                    help="天底/天顶修正起始仰角(度, 需 --edge-fix): 仰角低于 -该值(天底)或高于 +该值(天顶)的两条带里, "
                         "深度小于 --nadir-near 视为误判, 由内侧有效像素外推; 默认 80=上下修正 80° 以外")
    ap.add_argument("--side-deg", type=float, default=80.0,
                    help="左右两侧修正(度, 需 --edge-fix): 经度 |θ| 超过该值的区域(hequirect 经度覆盖 -90°..+90°), "
                         "同样把过近值视为无效并由内侧有效像素外推; 默认 80=只修正左右各 80° 以外; 0=该方向关闭")
    ap.add_argument("--nadir-near", type=float, default=0.2,
                    help="天底/两侧区域内的近距下限(米): 模型常把脚下地面估成 5~20cm, 会渲染成指向视点的尖刺")
    ap.add_argument("--mask", default="",
                    help="mask 图像路径 (.jpg/.png 等); 黑色区域(灰度<--mask-thr)的深度视为无效, 由邻近有效像素外推填充."
                         "mask 是单眼尺寸, 直接应用到左眼."
                         "不影响 --depth-cache / depthnpy 原始 .npy 导出 (mask 只作用于视频打包阶段)")
    ap.add_argument("--mask-thr", type=int, default=32,
                    help="mask 黑色阈值(0..255): 像素灰度 < 该值视为无效 (默认 32)")
    args = ap.parse_args()

    use_video = bool(args.input_video)
    left_files: list[Path] = []
    if use_video:
        vw, vh, vfps, total = probe_video(args.input_video)
        if vw % 2 != 0:
            sys.exit(f"视频宽度不是偶数, 无法对半拆分: {vw}")
        W, H = vw // 2, vh
        if vfps > 0:
            args.fps = vfps
        i0 = max(0, int(round(args.start * args.fps)))
        if args.end < 0:
            if total <= 0:
                sys.exit("源视频缺少总帧数, 请用 --end 指定结束时间")
            i1 = total
        else:
            i1 = int(round(args.end * args.fps))
            if total > 0:
                i1 = min(total, i1)
        if i1 <= i0:
            sys.exit(f"时间范围为空: [{args.start}, {args.end}) 帧 {i0}..{i1}")
        src_desc = f"{Path(args.input_video).name} ({vw}x{vh} SBS)"
    else:
        fdir = Path(args.frames_dir)
        left_files = sorted((fdir / "L").glob("*.png"))
        right_files = sorted((fdir / "R").glob("*.png"))
        if not left_files or len(left_files) != len(right_files):
            sys.exit(f"帧数量不匹配或为空: L={len(left_files)} R={len(right_files)}")
        i0 = max(0, int(round(args.start * args.fps)))
        i1 = len(left_files) if args.end < 0 else min(len(left_files), int(round(args.end * args.fps)))
        if i1 <= i0:
            sys.exit(f"时间范围为空: [{args.start}, {args.end}) 帧 {i0}..{i1}")
        first = cv2.imread(str(left_files[i0]))
        if first is None:
            sys.exit("读取首帧失败")
        H, W = first.shape[:2]
        src_desc = str(fdir)

    in_w = round32(args.depth_size)
    in_h = round32(in_w * H / W)
    n = i1 - i0
    dual = args.pack == "gray10mkv"
    depth_only = args.pack == "gray10mp4"
    depth_npy = args.pack == "depthnpy"
    both = args.both_eyes
    if both and not (depth_only or depth_npy):
        sys.exit("--both-eyes 只支持 --pack gray10mp4 / depthnpy")
    if both and args.swap_eyes:
        sys.exit("--both-eyes 与 --swap-eyes 互斥 (--both-eyes 已含右眼推理)")
    if both and args.depth_from:
        sys.exit("--both-eyes 不支持 --depth-from (缓存只有单眼深度)")
    if depth_npy and not args.depth_cache:
        sys.exit("depthnpy 模式必须指定 --depth-cache (深度 .npy 输出目录)")
    out_path = Path(args.out)
    depth_out = (out_path.with_name(out_path.stem + "_depth.mp4")   # depth_only/gray10mp4 的深度文件
                 if args.out else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # json 里的彩色视频: 默认原版 SBS (播放端裁左半); --color-file 可指 GUI 导出的单目 8bit 视频
    color_src = args.color_file or args.input_video
    color_name = Path(color_src).name if color_src else ""
    color_layout = args.color_layout or ("mono" if args.color_file else "sbs_left")
    print(f"[input] {src_desc}  单眼 {W}x{H}  帧 {i0}..{i1-1} (共 {n} 帧, {n/args.fps:.2f} s)")
    print(f"[model] Stereo Any Video  depth-size={in_w}x{in_h}  window={args.window} iters={args.iters}")
    if args.swap_eyes:
        print("[eye]   右眼基准推理 (水平翻转+交换左右眼, 深度=到右眼距离)")
    if depth_only:
        print(f"[output] {depth_out}  深度 {args.depth_bits}bit(无损) "
              f"{2 * in_w if both else in_w}x{in_h} (模型输入分辨率{'x2 左右眼拼接' if both else ''}) "
              f"+ {out_path.with_suffix('.json')}")
        print(f"[color]  颜色不导出, 直接使用: {color_name or '(未指定 --input-video)'}  "
              f"布局 {color_layout}  (mono=已裁单目, sbs_left=播放端取左半)")
    if depth_npy:
        where = (f"{args.depth_cache} (L/ 与 R/ 各一眼)" if both
                 else args.depth_cache)
        print(f"[output] {where}  深度缓存 .npy (float32 原始毫米, 未钳制, "
              f"{in_w}x{in_h})  不生成视频")
    if dual or depth_only:
        max_code = (1 << args.depth_bits) - 1
        if args.depth_encode == "log":
            print(f"[pack]   {args.pack}  深度 {args.depth_bits}bit 灰度 = log(depth/{args.min_range}) / "
                  f"log({args.max_range}/{args.min_range}) * {max_code}")
        else:
            print(f"[pack]   {args.pack}  深度 {args.depth_bits}bit 灰度 = (depth - {args.min_range}) / "
                  f"({args.max_range} - {args.min_range}) * {max_code}")
    elif dual:
        print(f"[output] {args.out}  MKV 双轨: 轨0 {W}x{H} yuv420p 8bit 彩色 + "
              f"轨1 {W}x{H} yuv420p10le 10bit 深度(无损)  crf={args.crf}")
    elif not depth_npy:
        print(f"[output] {args.out}  {2*W}x{H}  H.265 yuv420p10le crf={args.crf}")
        print(f"[pack]   gray10  右半幅 10bit 灰度 = (depth - {args.min_range}) / "
              f"({args.max_range} - {args.min_range}) * 1023")
    _edge_parts = []
    if args.edge_fix:
        if args.nadir_deg > 0:
            _edge_parts.append(f"上下|仰角|>{args.nadir_deg:g}°")
        if args.side_deg > 0:
            _edge_parts.append(f"两侧|经度|>{args.side_deg:g}°")
    _edge_desc = ("开 " + " ".join(_edge_parts) + f" 且 <{args.nadir_near:g}m 外推") if _edge_parts else "关闭"
    print(f"[filter] 视差域填充 <= {args.fill_max} px  深度域填充 "
          f"{'无限制' if args.fill_depth_dist <= 0 else f'<= {args.fill_depth_dist} px'}  "
          f"边缘修正 {_edge_desc}")

    def build_cmd(encoder: str) -> list[str]:
        # 深度输出尺寸 = 模型输入分辨率 (in_w x in_h), 双目拼接时宽度 x2; 只导深度时按该尺寸送入 ffmpeg
        dw = 2 * in_w if both else in_w
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # gray10mp4 只送深度 gray 平面 (dw 宽, 码值直通); 其余模式 hstack 左彩右深 (2W 宽 rgb48)
            "-f", "rawvideo", "-pix_fmt", gray_in_pix(args.depth_bits) if depth_only else "rgb48le",
            "-s", f"{dw}x{in_h}" if depth_only else f"{2*W}x{H}",
            "-r", str(args.fps), "-i", "-",
        ]
        if depth_only:
            # 只导深度: 颜色直接用原版 SBS, 播放端按宽高比自动裁左半
            # 深度已在 write_frame 中缩放到目标尺寸, 这里不再 scale
            if encoder == "hevc_nvenc":
                cmd += ["-c:v", "hevc_nvenc", "-preset", args.preset,
                        "-pix_fmt", "yuv420p" if args.depth_bits <= 8 else "p010le", "-lossless", "1"]
            else:
                cmd += ["-c:v", "libx265", "-preset", args.preset,
                        "-pix_fmt", yuv_out_pix(args.depth_bits),
                        # 深度恒无损 + 关 B 帧: 码值即精度, B 帧会拖慢解码顺序
                        "-x265-params", "log-level=error:range=full:lossless=1:bframes=0"]
            # 注意: 灰度深度数据不能带 -colorspace/-primaries/-trc 旗标,
            # 否则 swscale gray->yuv420pNle 走矩阵路径引入 +0.7% 相对增益失真 (实测阶梯图)
            cmd += ["-color_range", "pc", "-tag:v", "hvc1",
                    "-movflags", "+faststart", str(depth_out)]
            return cmd
        if dual:
            # 单进程拆两路: 左半 8bit 彩色轨 + 右半 10bit 深度轨写入同一容器,
            # 同一输入源、同帧数同时基, 天然同步
            cmd += [
                "-filter_complex",
                f"[0:v]crop={W}:{H}:0:0,scale=out_color_matrix=bt709:out_range=full[c];"
                f"[0:v]crop={W}:{H}:{W}:0,scale=out_color_matrix=bt709:out_range=full[d]",
                "-map", "[c]", "-map", "[d]",
            ]
            if encoder == "hevc_nvenc":
                cmd += ["-c:v:0", "hevc_nvenc", "-preset:v:0", args.preset,
                        "-cq:v:0", str(args.crf), "-pix_fmt:v:0", "yuv420p",
                        "-c:v:1", "hevc_nvenc", "-preset:v:1", args.preset,
                        "-pix_fmt:v:1", "p010le", "-lossless:v:1", "1"]
            else:
                cmd += ["-c:v:0", "libx265", "-preset:v:0", args.preset,
                        "-crf:v:0", str(args.crf), "-pix_fmt:v:0", "yuv420p",
                        "-x265-params:v:0", "log-level=error:range=full",
                        "-c:v:1", "libx265", "-preset:v:1", args.preset,
                        "-pix_fmt:v:1", "yuv420p10le",
                        # 深度轨恒无损 + 关 B 帧: 码值即精度, B 帧会拖慢解码顺序
                        "-x265-params:v:1", "log-level=error:range=full:lossless=1:bframes=0"]
            cmd += ["-color_range", "pc", "-colorspace", "bt709",
                    "-color_primaries", "bt709", "-color_trc", "bt709",
                    # mp4 里 HEVC 默认 tag hev1, hvc1 兼容性更好
                    "-tag:v:0", "hvc1", "-tag:v:1", "hvc1"]
            cmd += [str(out_path)]
            return cmd
        # 显式 bt709 全范围: 不指定时 swscale 默认 bt601, 浏览器对 4K 按 bt709
        # 解码会偏色; 灰度 (R=G=B) 与矩阵无关, 深度半幅不受影响
        cmd += ["-vf", "scale=out_color_matrix=bt709:out_range=full"]
        if encoder == "hevc_nvenc":
            # nvenc 无 -crf, 用 -cq; 10bit 4:2:0 走 p010le (yuv420p10le 的左对齐形式)
            cmd += ["-c:v", "hevc_nvenc", "-preset", args.preset, "-cq", str(args.crf),
                    "-pix_fmt", "p010le"]
        else:
            cmd += ["-c:v", "libx265", "-preset", args.preset, "-crf", str(args.crf),
                    "-pix_fmt", "yuv420p10le",
                    "-x265-params", "log-level=error:range=full"]
        cmd += ["-color_range", "pc", "-colorspace", "bt709",
                "-color_primaries", "bt709", "-color_trc", "bt709",
                "-tag:v", "hvc1", "-movflags", "+faststart",
                str(out_path)]
        return cmd

    def start_encoder() -> subprocess.Popen:
        encs = ["libx265", "hevc_nvenc"] if args.encoder == "auto" else [args.encoder]
        if depth_only and args.depth_bits == 12 and "hevc_nvenc" in encs:
            print("[encoder] hevc_nvenc 不支持 12bit, 仅用 libx265")
            encs = ["libx265"]
        for enc in encs:
            # stderr 直接继承到控制台: 编码器中途死亡时能看到真实原因,
            # 且避免 stderr 管道缓冲(64KB)写满导致编码器卡死
            p = subprocess.Popen(build_cmd(enc), stdin=subprocess.PIPE, startupinfo=NO_WINDOW)
            if p.stdin is None:
                continue
            # 给编码器一个启动窗口: 有些驱动一启动就报错退出
            time.sleep(1.0)
            if p.poll() is None:
                print(f"[encoder] {enc}")
                return p
            print(f"[encoder] {enc} 启动失败, 尝试下一个")
        sys.exit("ffmpeg 启动失败")

    proc = None if depth_npy else start_encoder()

    model = None
    if not args.depth_from:
        model = SAVModel(args.model, args.vda, amp=not args.no_amp)
    else:
        print(f"[cache] 从 {args.depth_from} 读取深度缓存 (.npy), 跳过模型")
    if args.depth_cache:
        Path(args.depth_cache).mkdir(parents=True, exist_ok=True)
        if depth_npy and both:
            for eye in ("L", "R"):
                (Path(args.depth_cache) / eye).mkdir(parents=True, exist_ok=True)
    left_cache: dict[int, np.ndarray] = {}
    written = 0
    t_start = time.perf_counter()

    # 边缘修正(--edge-fix, 默认开启): 只修正经度 |θ| 超过 --side-deg(默认 75°) 的左右边缘;
    # 上下天底/天顶带默认不修正(--nadir-deg=0), 设为非 0 时 |仰角| > nadir-deg 的上下两条带都会修正.
    # 两侧水平视差不可靠(接近与基线共线), 模型会把它们估成 5~20cm,
    # 顶点半径缩到视点附近就成了指向用户的尖刺/近墙.
    # 开启后把区域内"过近"的像素视为无效, 交给最近邻填充从邻近有效像素外推, 左右由内侧画面延续, 不再有尖刺.
    edge_cache: dict[tuple, np.ndarray] = {}

    # ---- mask 加载 (可选, 用于把黑色区域深度视为无效; 只影响打包阶段, 不影响 depth_npy) ----
    def _load_mask(path: str, out_h: int, out_w: int) -> np.ndarray | None:
        if not path:
            return None
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            sys.exit(f"mask 读取失败: {path}")
        mh, mw = img.shape[:2]
        thresh = int(args.mask_thr)
        # SBS mask 判定: 宽度约等于 2 * 单眼宽 (容忍 2px 取整误差)
        if abs(mw - 2 * out_w) <= 2:
            img = cv2.resize(img, (2 * out_w, out_h), interpolation=cv2.INTER_AREA)
            left = cv2.resize(img[:, :out_w], (out_w, out_h), interpolation=cv2.INTER_AREA)
            black = left < thresh
            print(f"[mask] {Path(path).name} SBS 布局 {2*out_w}x{out_h} → 左眼 {out_w}x{out_h}"
                  f"  黑色区域占比 {black.mean()*100:.1f}%  (阈值 {thresh})")
            return black
        # 单眼 mask
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        black = img < thresh
        print(f"[mask] {Path(path).name} 单眼 {out_w}x{out_h}  黑色区域占比 {black.mean()*100:.1f}%  (阈值 {thresh})")
        return black

    mask_black = _load_mask(args.mask, in_h, in_w)   # (in_h, in_w) bool, True=无效

    def edge_mask(shape) -> np.ndarray:
        """不可信区域: |仰角| > nadir-deg 的上下两条带 (equirect: phi = (0.5 - (y+0.5)/h) * 180°)
        与 |经度| > side-deg (theta = (x/W - 0.5) * 180°); 未开 --edge-fix 时全 False"""
        m = edge_cache.get(shape)
        if m is None:
            h, w = shape
            m = np.zeros(shape, bool)
            if args.edge_fix and args.nadir_deg > 0:
                # 天底(下方) + 天顶(上方) 双向修正: |仰角| > nadir-deg 的两条带
                rows = h * np.radians(args.nadir_deg) / np.pi
                b0 = int(np.ceil(h * 0.5 + rows - 0.5))       # 下方带起点
                t1 = int(np.floor(h * 0.5 - rows - 0.5)) + 1  # 上方带终点(不含)
                m[min(max(b0, 0), h):, :] = True
                m[:min(max(t1, 0), h), :] = True
            if args.edge_fix and args.side_deg > 0:
                k = int(np.floor(w * (0.5 - args.side_deg / 180.0)))
                m[:, :min(max(k, 0), w)] = True
                m[:, min(max(w - k, 0), w):] = True
            edge_cache[shape] = m
        return m

    def finalize_depth(d: np.ndarray) -> np.ndarray:
        """原始米制深度 -> 输出用深度: 无效点(含 mask 黑区、edge-fix 过近值)用最近邻填满, 再钳到 [near, far]"""
        bad = d <= 0
        band = edge_mask(d.shape)
        if band.any():
            bad = bad | (band & (d < args.nadir_near))
        if mask_black is not None:
            # mask 是 (in_h, in_w), d 当前也是模型输出分辨率 (in_h, in_w), 直接融合
            bad = bad | mask_black
        if bad.any() and not bad.all():
            d = nearest_fill(d, ~bad, args.fill_depth_dist)
        return np.clip(d, args.min_range, args.max_range)

    def write_frame(didx: int, disp_f: np.ndarray, disp_f_r: np.ndarray | None = None):
        nonlocal written
        left_img = left_cache.pop(didx, None)
        if left_img is None:
            sys.exit(f"帧 {didx} 左图缺失")
        if disp_f is None:
            # 深度缓存: 只支持 float32 .npy (原始毫米, 未钳制)
            npf = Path(args.depth_from) / f"{didx:05d}.npy"
            if not npf.is_file():
                sys.exit(f"深度缓存缺失: {npf} (仅支持 float32 .npy)")
            raw_mm = np.load(npf).astype(np.float32)
        else:
            depth_raw = disp_to_range_hequirect(disp_f, args.baseline,
                                                min_disp_px=MIN_DISP_PX,
                                                min_cos=args.min_cos)
            raw_mm = (depth_raw * 1000.0).astype(np.float32)   # 原始毫米, 不钳制
            if args.depth_cache and not depth_npy:
                # 缓存原始毫米: 之后换 near/far 无需重新推理
                np.save(Path(args.depth_cache) / f"{didx:05d}.npy", raw_mm)
        # 右眼深度 (--both-eyes): 同样换算成原始毫米, 打包时与左眼左右拼接
        raw_mm_r = None
        if disp_f_r is not None:
            depth_raw_r = disp_to_range_hequirect(disp_f_r, args.baseline,
                                                  min_disp_px=MIN_DISP_PX,
                                                  min_cos=args.min_cos)
            raw_mm_r = (depth_raw_r * 1000.0).astype(np.float32)
        if depth_npy:
            # 仅导出深度缓存: float32 原始毫米 .npy, 可直接喂给 --depth-from
            root = Path(args.depth_cache)
            if both:
                # 双目: 左右眼各一个子目录, 文件名序号相同
                np.save(root / "L" / f"{didx:05d}.npy", raw_mm)
                np.save(root / "R" / f"{didx:05d}.npy", raw_mm_r)
            else:
                np.save(root / f"{didx:05d}.npy", raw_mm)
            written += 1
            return
        # 钳制只在输出阶段做: 无效点交给最近邻填充, 远处钳到 max_range
        depth_small = finalize_depth(raw_mm.astype(np.float32) / 1000.0)
        depth_small_r = (finalize_depth(raw_mm_r.astype(np.float32) / 1000.0)
                         if raw_mm_r is not None else None)
        try:
            # 目标尺寸: 深度输出固定为模型输入分辨率; 颜色混流模式必须与单眼原图同尺寸
            if depth_only:
                tgt_w, tgt_h = in_w, in_h
            else:
                tgt_w, tgt_h = W, H
            # 默认导出分辨率与模型输入分辨率一致, 无需缩放; 尺寸不同时才缩放并补填充
            if tgt_w == depth_small.shape[1] and tgt_h == depth_small.shape[0]:
                depth_full = depth_small
            else:
                depth_full = resize_depth(depth_small, tgt_w, tgt_h)
            d16 = encode_depth_gray10(depth_full, args.min_range, args.max_range, tgt_w,
                                      args.depth_encode, args.depth_bits)
            if depth_small_r is not None:
                depth_full_r = (depth_small_r if (tgt_w == depth_small_r.shape[1]
                                                  and tgt_h == depth_small_r.shape[0])
                                else resize_depth(depth_small_r, tgt_w, tgt_h))
                d16_r = encode_depth_gray10(depth_full_r, args.min_range, args.max_range,
                                            tgt_w, args.depth_encode, args.depth_bits)
                d16 = np.hstack([d16, d16_r])   # 深度帧左右拼接: 左半=左眼, 右半=右眼
            # gray10mp4 只送深度 gray 平面 (tgt_w 宽, 码值原样); 其余模式 hstack 左彩右深 (2W 宽 rgb48)
            if depth_only:
                frame = np.ascontiguousarray(d16 if args.depth_bits > 8 else d16.astype(np.uint8))
            else:
                # cv2 是 BGR, rgb48le 需要 R,G,B 顺序; 8bit -> 16bit 用 *257 精确展开
                code16 = (d16 << (16 - args.depth_bits)).astype(np.uint16)
                left16 = left_img[:, :, ::-1].astype(np.uint16) * 257
                frame = np.ascontiguousarray(
                    np.hstack([left16, np.repeat(code16[:, :, None], 3, axis=2)])).astype("<u2")
            # 单帧 176MB, 一次 write 在 Windows 管道上可能 EINVAL, 分 32MB 块写入
            mv = memoryview(frame.data)
            for off in range(0, mv.nbytes, 32 * 1024 * 1024):
                proc.stdin.write(mv[off:off + 32 * 1024 * 1024])
            proc.stdin.flush()
        except BrokenPipeError:
            sys.exit("ffmpeg 提前退出 (检查磁盘空间/编码器)")
        written += 1

    def frame_source():
        if use_video:
            for k, (l, r) in enumerate(iter_sbs_frames(args.input_video, vw, vh, i0, n, args.fps)):
                yield i0 + k, l, r
        else:
            for idx in range(i0, i1):
                l = cv2.imread(str(left_files[idx]), cv2.IMREAD_COLOR)
                r = cv2.imread(str(right_files[idx]), cv2.IMREAD_COLOR)
                if l is None or r is None:
                    sys.exit(f"读取第 {idx} 帧失败")
                yield idx, l, r

    src = iter(frame_source())
    src_done = False
    buf: deque[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = deque()

    def ensure(up_to: int):
        nonlocal src_done
        while not src_done and (not buf or buf[-1][0] < up_to):
            try:
                idx, l, r = next(src)
            except StopIteration:
                src_done = True
                break
            left_cache[idx] = l
            ls = cv2.resize(l, (in_w, in_h), interpolation=cv2.INTER_AREA)
            rs = cv2.resize(r, (in_w, in_h), interpolation=cv2.INTER_AREA)
            if args.swap_eyes:
                # 右眼基准: 水平翻转 + 交换 (镜像世界里视差重新为正), 视差结果翻回即右眼坐标
                ls, rs = cv2.flip(rs, 1), cv2.flip(ls, 1)
            buf.append((idx - i0, l, ls, rs))

    if args.depth_from:
        # 跳过推理: 逐帧读左图 + 缓存深度, 直接打包编码
        for idx in range(i0, i0 + n):
            ensure(idx)
            buf.clear()
            write_frame(idx, None)
            done = idx - i0 + 1
            print(f"\r[{done}/{n}] 合成中", end="", flush=True)
    else:
        plan = window_plan(n, args.window, overlap=max(1, args.window // 4))
        emitted = 0
        for s, e, keep_end in plan:
            ensure(e - 1)
            win = [it for it in buf if s <= it[0] < e]
            if len(win) != e - s:
                sys.exit(f"窗口帧缺失: 需要 {s}..{e-1}, 实际 {len(win)}")
            disp = model.infer_window([it[2] for it in win], [it[3] for it in win],
                                      in_w, in_h, args.iters)
            if args.swap_eyes:
                disp = disp[:, :, ::-1]     # 翻回右眼图像坐标, 与视频右半幅逐像素对齐
            disp_r = None
            if both:
                # 右眼: 水平翻转+交换送入 (镜像世界视差为正), 视差翻回右眼坐标系
                disp_r = model.infer_window([cv2.flip(it[3], 1) for it in win],
                                            [cv2.flip(it[2], 1) for it in win],
                                            in_w, in_h, args.iters)[:, :, ::-1]
            for idx in range(emitted, keep_end):
                write_frame(i0 + idx, disp[idx - s],
                            None if disp_r is None else disp_r[idx - s])
            emitted = keep_end
            cutoff = max(0, emitted - max(1, args.window // 4))
            while buf and buf[0][0] < cutoff:
                buf.popleft()

            done = emitted
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (n - done) if done and n > done else 0.0
            print(f"\r[{done}/{n}] 已用 {elapsed/60:.1f} min  剩余约 {eta/60:.1f} min",
                  end="", flush=True)

    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            sys.exit("ffmpeg 编码失败")

    if depth_npy:
        dw, dh = in_w, in_h
        meta = {"near": args.min_range, "far": args.max_range,
                "encode": "mm_f32", "format": "npy", "bit_depth": 32,
                "unit": "millimeter", "clamped": False,
                "eye": "both" if both else ("right" if args.swap_eyes else "left"),
                "frames": written,
                "color_file": color_name, "color_layout": color_layout}
        if color_layout != "mono":      # mono 彩色视频已由 GUI 裁剪成片段, 播放端不再偏移
            meta.update(time_range_fields(args.start, args.end))
        meta_path = Path(args.depth_cache) / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"\n[done] {Path(args.depth_cache).resolve()}  {written} 帧 .npy "
              f"(float32 原始毫米, 未钳制, {dw}x{dh})")
        print(f"[done] {meta_path.resolve()}")
        return
    if depth_only:
        # 深度分辨率固定为模型输入分辨率: 播放端用 ffprobe 读深度视频实际尺寸, json 不再记录 width/height
        meta = {"near": args.min_range, "far": args.max_range,
                "encode": args.depth_encode, "bits": args.depth_bits,
                # 颜色不导出: color_file 指彩色视频 (默认原版 SBS, 播放端裁左半; 可用 --color-file 指单目视频)
                "color_file": color_name,
                "color_layout": color_layout,
                "depth_file": depth_out.name}
        if both:
            # 深度视频左右拼接: 左半=左眼深度, 右半=右眼深度 (缺省=单左目, 颜色/深度都裁左半)
            meta["depth_layout"] = "sbs"
    else:
        meta = {"near": args.min_range, "far": args.max_range, "pack": args.pack,
                "encode": args.depth_encode, "width": 2 * W, "height": H}
    if dual:
        meta.update({"eye_width": W, "tracks": 2,
                     "track0": "color yuv420p 8bit", "track1": "depth yuv420p10le 10bit"})
    if color_layout != "mono":          # mono 彩色视频已由 GUI 裁剪成片段, 播放端不再偏移
        meta.update(time_range_fields(args.start, args.end))
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if depth_only:
        print(f"\n[done] {depth_out.resolve()}  {depth_out.stat().st_size/1e6:.1f} MB  {written} 帧")
        print(f"[done] {out_path.with_suffix('.json').resolve()}")
        if args.input_video:
            print(f"[done] 颜色 = {color_name}  ({color_layout})")
    else:
        print(f"\n[done] {out_path.resolve()}  {out_path.stat().st_size/1e6:.1f} MB  {written} 帧")
    if dual:
        print(f"[godot] 用 --src {out_path.resolve()} 打开, 播放器自动按双轨解码")
    elif not depth_only:
        print(f"[url]  /vr_player/video.html?src=/{out_path.as_posix()}"
              f"&near={args.min_range}&far={args.max_range}&pack=gray10&encode=linear")


if __name__ == "__main__":
    main()
