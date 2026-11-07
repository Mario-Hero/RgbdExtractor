# -*- coding: utf-8 -*-
"""VR180 深度视频导出 GUI (tkinter)

两个选项卡:
  使用视频进行推理  -> 调用模型推理深度, 可设模型输入分辨率/滑动窗口, 显示显存估算
  使用已有的深度图  -> 读取 --depth-from 目录下的 float32 .npy 原始毫米深度缓存, 跳过推理

包装 process_video.py (--pack gray10mp4), 颜色不导出 (默认用原版 SBS, 播放端裁左半), 输出:
  <名字>_depth.mp4   8/10/12bit 深度视频 (HEVC 无损, 分辨率 = 模型输入分辨率, 播放端自动识别位深)
  <名字>.json        播放端元数据 (near/far/encode/bits/原 SBS 文件名)

SBS 输入框可直接拖入视频或文件夹 (文件夹仅“使用视频进行推理”选项卡支持, 会按顺序导出其中所有视频,
跳过已导出的 _mono8; “使用已有的深度图”选项卡只接受单个视频文件)。
  输入框下方勾选「导出单目 8bit 视频」: 点“开始导出”时先裁左半幅转 HEVC 8bit (<名字>_mono8.mp4),
  再导深度; 播放端不必再解 8K SBS, json 的 color_file 指向它、color_layout=mono

另: “仅导出深度缓存”把 float32 原始毫米 .npy 写进 <名字>_depth_npy/ (未钳制, 可直接作为
  “使用已有的深度图”的输入, 之后换 near/far 只需重跑打包, 无需重新推理)

用法: python gui_export.pyw [--lang zh|en|auto]   (默认为 auto, 跟随系统语言)
"""

import json
import math
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

try:  # 拖放支持: 缺 tkinterdnd2 时退化为普通 Tk (只是不能往输入框里拖文件)
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES, TkinterDnD = None, None

try:  # 图像预览: 缺 Pillow 时静默跳过 (mask 功能仍可用, 只是没预览)
    from PIL import Image, ImageTk

    _HAS_PIL = True
except Exception:
    Image = ImageTk = None
    _HAS_PIL = False

ROOT = Path(__file__).resolve().parent
PROCESS = ROOT / "process_video.py"

PAD = {"padx": 8, "pady": 4}

# Windows: 子进程 (python 及它再拉起的 ffmpeg) 不弹控制台窗口
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0

# log 模式展示这些距离点的量化步长
LOG_DEMO_DISTANCES = (0.25, 0.5, 1.0, 2.0)

# 单目彩色视频后缀: <SBS 名字>_mono8.mp4 (左半幅 8bit HEVC, 播放端不用再解 8K SBS)
MONO_SUFFIX = "_mono8"

# SBS 输入框接受拖放的文件类型
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}

# mask 预览缩略图尺寸 (px, PIL Image.thumbnail 保持比例)
MASK_PREVIEW_SIZE = (96, 96)

# 显存估算标定: depth-size=640 / window=20 约 12GB
VRAM_BASE_GB = 2.0  # 模型常驻 (主权重 + VDA)
VRAM_PER_PX_FRAME = 1.22e-6  # GB / (像素 * 帧)


def detect_lang() -> str:
    """按系统界面语言选 zh / en; Windows 读 GetUserDefaultUILanguage, 其它系统看 LANG 环境变量"""
    try:
        if os.name == "nt":
            import ctypes

            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()
            # 低 10 位是主语言 ID: 0x04 = LANG_CHINESE
            return "zh" if (langid & 0x3FF) == 0x04 else "en"
    except Exception:
        pass
    for key in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        v = os.environ.get(key, "").strip().lower()
        if v.startswith("zh"):
            return "zh"
        if v:
            return "en"
    return "zh"


def lang_from_args(argv) -> str | None:
    """从命令行取语言覆盖: --lang zh|en|auto (也接受 -l)。auto / 非法值返回 None 表示跟随系统"""
    for i, a in enumerate(argv):
        v = None
        if a in ("--lang", "-l"):
            v = argv[i + 1] if i + 1 < len(argv) else ""
        elif a.startswith("--lang="):
            v = a.split("=", 1)[1]
        if v is None:
            continue
        v = v.strip().lower()
        if v in ("zh", "en"):
            return v
        if v in ("auto", ""):
            return None
    return None


LANG = lang_from_args(sys.argv[1:]) or detect_lang()

TEXT = {
    "title": {"zh": "VR180 深度视频导出", "en": "VR180 Depth Video Export"},
    "tab.infer": {"zh": "使用视频进行推理", "en": "Infer from video"},
    "tab.depth": {"zh": "使用已有的深度图", "en": "Use existing depth maps"},
    "log.title": {"zh": "日志", "en": "Log"},
    "browse": {"zh": "浏览…", "en": "Browse…"},
    "start": {"zh": "开始导出", "en": "Start export"},
    "cancel": {"zh": "取消导出", "en": "Cancel export"},
    "sbs.infer.label": {
        "zh": "SBS 视频输入 (双目文件或文件夹, 可直接拖进来):",
        "en": "SBS input (stereo file or folder, drag & drop supported):",
    },
    "sbs.depth.label": {
        "zh": "SBS 视频输入 (单个双目文件, 可直接拖进来):",
        "en": "SBS input (single stereo file, drag & drop supported):",
    },
    "mono.check": {
        "zh": "导出起始到结束时间的左半幅单目 8bit HEVC 视频 (播放端不必再解 SBS。仅在解码压力过大时使用)",
        "en": "Export a left-eye mono 8-bit HEVC clip for the time range (the player then skips SBS decoding; use only when decoding is too heavy)",
    },
    "res.label": {
        "zh": "模型输入分辨率 (宽, --depth-size):",
        "en": "Model input width (--depth-size):",
    },
    "res.hint": {
        "zh": "px   (越大越清晰, 显存越大)",
        "en": "px   (larger is sharper but uses more VRAM)",
    },
    "win.label": {
        "zh": "滑动窗口大小 (帧, --window):",
        "en": "Sliding window (frames, --window):",
    },
    "win.hint": {
        "zh": "帧   (推荐 20, 越大时间一致性越好、显存越大)",
        "en": "frames   (20 recommended; larger improves temporal consistency and costs VRAM)",
    },
    "depth_only": {
        "zh": "仅导出深度缓存 (.npy) 到输出目录",
        "en": "Export depth cache (.npy) only to the output folder",
    },
    "both_eyes": {
        "zh": "双目深度: 左右眼各推理一次 (视频模式深度左右拼成 SBS, 缓存模式写 L/R 子目录; 推理时间约 x2)",
        "en": "Stereo depth: infer both eyes (video mode packs depth as SBS, cache mode writes L/R subfolders; about 2x)",
    },
    "depth.dir.label": {
        "zh": "帧深度目录 (float32 .npy, 原始毫米, 未钳制):",
        "en": "Per-frame depth folder (float32 .npy, raw millimetres, unclamped):",
    },
    "depth.dir.hint": {
        "zh": "读取该目录下 00000.npy、00001.npy… 并跳过模型推理",
        "en": "Reads 00000.npy, 00001.npy… from that folder and skips model inference",
    },
    "range.start": {"zh": "起始:", "en": "Start:"},
    "range.end": {"zh": "结束:", "en": "End:"},
    "range.hint": {
        "zh": "(纯秒数 或 MM:SS 或 HH:MM:SS; 结束留空 = 到末尾)",
        "en": "(seconds, MM:SS or HH:MM:SS; leave end empty = to the end)",
    },
    "near.label": {"zh": "near (米):", "en": "near (m):"},
    "far.label": {"zh": "far (米):", "en": "far (m):"},
    "mode.log": {
        "zh": "对数 log (近处更精细, 推荐)",
        "en": "Log (finer near the camera, recommended)",
    },
    "mode.linear": {"zh": "线性 linear (误差恒定)", "en": "Linear (constant error)"},
    "bits.label": {"zh": "位深:", "en": "Bit depth:"},
    "bits.8": {"zh": "8bit (255 级)", "en": "8-bit (255 levels)"},
    "bits.10": {"zh": "10bit (1023 级)", "en": "10-bit (1023 levels)"},
    "bits.12": {"zh": "12bit (4095 级)", "en": "12-bit (4095 levels)"},
    "res.note": {
        "zh": "深度视频分辨率 = 模型输入分辨率 (上方设置)",
        "en": "Depth video resolution equals the model input width (set above)",
    },
    "out.label": {"zh": "输出目录:", "en": "Output folder:"},
    "same_dir": {
        "zh": "导出到输入视频相同目录",
        "en": "Export next to the input video",
    },
    "mask.frame": {
        "zh": "可选: mask (黑色区域深度无效, 打包时外推填充, 不影响.npy原始导出)",
        "en": "Optional: mask (black areas get no depth and are extrapolated when packing; raw .npy export is unaffected)",
    },
    "mask.label": {"zh": "mask 图像:", "en": "Mask image:"},
    "mask.preview": {"zh": "(预览)", "en": "(preview)"},
    "mask.load_fail": {"zh": "(加载失败)", "en": "(load failed)"},
    "mask.pil_hint": {
        "zh": "(安装 Pillow 可显示 mask 预览: pip install Pillow)",
        "en": "(install Pillow to show the mask preview: pip install Pillow)",
    },
    "mask.pick_title": {
        "zh": "选择 mask 图像 (黑色区域 = 深度无效)",
        "en": "Choose a mask image (black = no depth)",
    },
    "filter.images": {"zh": "图像", "en": "Images"},
    "filter.videos": {"zh": "视频", "en": "Videos"},
    "filter.all": {"zh": "所有文件", "en": "All files"},
    "dlg.video": {"zh": "选择 SBS 视频", "en": "Choose an SBS video"},
    "dlg.depthdir": {
        "zh": "选择帧深度目录 (float32 .npy, 原始毫米)",
        "en": "Choose the per-frame depth folder (float32 .npy, raw millimetres)",
    },
    "dlg.outdir": {"zh": "选择输出目录", "en": "Choose the output folder"},
    "err.title": {"zh": "错误", "en": "Error"},
    "err.input": {
        "zh": "请选择有效的视频输入 (文件或包含视频的文件夹)",
        "en": "Pick a valid video input (a file or a folder containing videos)",
    },
    "err.input_file": {
        "zh": "请选择有效的视频输入文件",
        "en": "Pick a valid video input file",
    },
    "err.near_far": {"zh": "需要 0 < near < far", "en": "needs 0 < near < far"},
    "err.time": {
        "zh": "起始/结束时间需为秒数 (如 90.5) 或 MM:SS (如 1:30) 或 HH:MM:SS (如 1:02:03),\n"
        "且结束 > 起始 (结束留空 = 到末尾)",
        "en": "Start/end times must be seconds (e.g. 90.5), MM:SS (e.g. 1:30) or HH:MM:SS (e.g. 1:02:03),\n"
        "and end must be greater than start (leave end empty = to the end)",
    },
    "err.out": {"zh": "输出目录无效", "en": "Invalid output folder"},
    "err.res_window": {
        "zh": "模型输入分辨率需 >= 32, 滑动窗口需 >= 4",
        "en": "Model width must be >= 32 and the window >= 4",
    },
    "err.depthdir": {
        "zh": "帧深度目录不存在",
        "en": "The per-frame depth folder does not exist",
    },
    "prec.need": {"zh": "需要 0 < near < far", "en": "needs 0 < near < far"},
    "prec.log_rel": {
        "zh": "对数 log: 相对误差恒定 {pct:.2f}%",
        "en": "Log: constant relative error {pct:.2f}%",
    },
    "prec.log_near": {
        "zh": "近处量化细, 远处粗 (人眼不敏感)",
        "en": "fine near the camera, coarse far away (the eye is not sensitive there)",
    },
    "prec.linear": {
        "zh": "线性 linear: 绝对误差全程恒定 {mm}",
        "en": "Linear: constant absolute error {mm}",
    },
    "prec.linear_same": {
        "zh": "近处与远处精度相同",
        "en": "same precision near and far",
    },
    "vram.bad": {
        "zh": "显存估算: 模型输入分辨率/滑动窗口需为有效整数",
        "en": "VRAM estimate: model width / window must be valid integers",
    },
    "vram.est": {
        "zh": "显存估算: 约 {gb:.1f} GB  (模型常驻 ~2GB + 激活 ∝ 分辨率²×窗口, 单眼按 1:1)",
        "en": "VRAM estimate: about {gb:.1f} GB  (model ~2 GB resident + activations ∝ width² × window, eye assumed 1:1)",
    },
    "log.batch_n": {"zh": "[批量] 共 {n} 个视频", "en": "[batch] {n} videos"},
    "log.cancel_stop": {"zh": "[取消] 正在停止…", "en": "[cancel] stopping…"},
    "log.cancel_done": {"zh": "[取消] 导出已停止", "en": "[cancel] export stopped"},
    "log.cancel_kill": {
        "zh": "[取消] 终止进程失败: {e}",
        "en": "[cancel] failed to kill the process: {e}",
    },
    "log.skip_mono": {
        "zh": "[跳过] 单目视频已存在: {p}",
        "en": "[skip] mono clip already exists: {p}",
    },
    "log.mono": {"zh": "单目: {name}", "en": "mono: {name}"},
    "log.mono_fail": {
        "zh": "[失败] 单目视频导出失败 (退出码 {code}), 跳过该视频",
        "en": "[failed] mono clip export failed (exit {code}); skipping this video",
    },
    "log.mono_done": {
        "zh": "[完成] 单目视频: {p}  ({mb:.1f} MB, 左半幅 8bit HEVC)",
        "en": "[done] mono clip: {p}  ({mb:.1f} MB, left-eye 8-bit HEVC)",
    },
    "log.json_patched": {
        "zh": "已更新 {stem}.json: color_file={name}, color_layout=mono",
        "en": "updated {stem}.json: color_file={name}, color_layout=mono",
    },
    "log.mask_preview_fail": {
        "zh": "[mask] 预览失败: {e}",
        "en": "[mask] preview failed: {e}",
    },
    "log.npy_done": {
        "zh": "[完成] {p}  ({n} 帧 .npy, float32 原始毫米)",
        "en": "[done] {p}  ({n} frames of .npy, float32 raw millimetres)",
    },
    "log.video_done": {"zh": "[完成] 导出:", "en": "[done] exported:"},
    "log.depth_fail": {
        "zh": "[失败] 导出失败 (退出码 {code})",
        "en": "[failed] export failed (exit {code})",
    },
    "log.exception": {"zh": "[失败] 异常: {e}", "en": "[failed] exception: {e}"},
    "log.batch_end": {
        "zh": "[批量] 结束: 成功 {ok} / 共 {n}",
        "en": "[batch] finished: {ok} of {n} succeeded",
    },
    "log.drop": {
        "zh": "[拖入] {p}{extra}",
        "en": "[drop] {p}{extra}",
    },
    "log.drop_dir_extra": {"zh": " (目录下 {n} 个视频)", "en": " ({n} videos in the folder)"},
    "log.drop_ignored": {
        "zh": "[忽略] 该选项卡只支持单个视频文件, 不支持文件夹: {p}",
        "en": "[ignored] this tab accepts a single video file, not a folder: {p}",
    },
}


def tr(key: str, **kw) -> str:
    """取当前语言的文案, kw 用于格式化占位符"""
    s = TEXT.get(key, {}).get(LANG) or TEXT.get(key, {}).get("en", key)
    return s.format(**kw) if kw else s


def fmt_mm(m: float) -> str:
    return f"{m * 1000:.2f} mm" if m < 0.01 else f"{m * 1000:.1f} mm"


def has_nvenc() -> bool:
    """ffmpeg 是否带 hevc_nvenc (N 卡硬编, 8K 裁半转码快很多)"""
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            creationflags=NO_WINDOW,
        )
        return "hevc_nvenc" in (r.stdout or "")
    except Exception:
        return False


def patch_color(meta_path: Path, color_name: str) -> bool:
    """把 json 的 color_file / color_layout 改为单目视频"""
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        data["color_file"] = color_name
        data["color_layout"] = "mono"
        meta_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def precision_text(near: float, far: float, mode: str, bits: int = 10) -> str:
    """编码模式的量化步长说明 (随 near/far/位深 动态更新)"""
    if near <= 0 or far <= near:
        return tr("prec.need")
    max_code = (1 << bits) - 1  # 8bit=255, 10bit=1023, 12bit=4095
    lines = []
    if mode == "log":
        rel = (
            math.log(far / near) / max_code
        )  # 相对误差 = ln(far/near)/max_code, d 处步长 = d*rel
        lines.append(tr("prec.log_rel", pct=rel * 100))
        parts = [f"{d:g}m → {fmt_mm(d * rel)}" for d in LOG_DEMO_DISTANCES]
        lines.append("  " + "   ".join(parts))
        lines.append("  " + tr("prec.log_near"))
    else:
        step = (far - near) / max_code
        lines.append(tr("prec.linear", mm=fmt_mm(step)))
        lines.append("  " + tr("prec.linear_same"))
    return "\n".join(lines)


def vram_estimate(width_px: int, window: int) -> float:
    """粗略显存估算 (GB): 模型常驻 + 激活(∝ 分辨率² × 窗口帧数)

    以 depth-size=640 / window=20 约 12GB 标定后线性外推, 仅供参考。
    假设单眼画面为 1:1 (VR180 SBS 视频 2:1)。
    """
    return VRAM_BASE_GB + width_px * width_px * window * VRAM_PER_PX_FRAME


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(tr("title"))
        root.resizable(False, False)
        self.log_q: queue.Queue[str] = queue.Queue()
        self.running = False
        self._cancel = threading.Event()  # 勾选后 worker 在下一个检查点退出
        self._proc = None  # 当前子进程 (取消时杀掉)
        self.btn_meta: dict = {}  # 按钮 -> (常规文字, 常规命令), 取消时临时改写
        self.cjk_family = self._pick_cjk_family()
        self._build()
        root.after(80, self._poll_log)

    # ---------- 字体 ----------

    def _pick_cjk_family(self) -> str:
        """挑选含中文字形的字体, 避免日志/提示中的中文显示为方块或乱码"""
        try:
            fams = set(tkfont.families(self.root))
        except tk.TclError:
            return ""
        for f in (
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "SimSun",
        ):
            if f in fams:
                return f
        return ""

    def _font(self, size: int = 9) -> dict:
        return {"font": (self.cjk_family, size)} if self.cjk_family else {}

    # ---------- 界面 ----------

    def _build(self):
        # 两个选项卡共享这些变量, 任一处修改另一处自动同步
        self.v_video = tk.StringVar()
        self.v_depthfrom = tk.StringVar()
        self.v_near = tk.StringVar(value="0.05")
        self.v_far = tk.StringVar(value="5.0")
        self.v_start = tk.StringVar(value="0")
        self.v_end = tk.StringVar()
        self.v_out = tk.StringVar(value=str(ROOT / "out"))
        self.v_depth_size = tk.StringVar(value="640")
        self.v_window = tk.StringVar(value="20")
        self.mode = tk.StringVar(value="log")
        self.bits = tk.IntVar(value=10)
        self.same_dir = tk.BooleanVar(value=False)
        self.v_mono_export = tk.BooleanVar(
            value=False
        )  # 勾选=导出时顺带裁左半幅单目 8bit 视频
        self.v_both_eyes = tk.BooleanVar(
            value=False
        )  # 勾选=左右眼各推理一次, 深度拼左右 SBS (播放器双球壳)
        self.v_mask = tk.StringVar()  # mask 图像 (可空, SBS 或单眼皆可)
        self._mask_prevs: list[tk.Label] = []  # 预览 Label 引用 (两个选项卡各一个)
        self._mask_img_ref = None  # PhotoImage 引用 (防 GC)
        self._mask_prev_job = None  # 预览防抖 after id
        # 只要 mask 路径被改动 (手输/浏览/自动补全), 立即刷新预览
        self.v_mask.trace_add("write", lambda *a: self._schedule_mask_preview())

        self.prec_labels: list[tk.Label] = []
        self.out_entries: list[ttk.Entry] = []
        self.video_entries: list[tuple[ttk.Entry, bool]] = (
            []
        )  # (SBS 输入框, 是否允许文件夹)
        self.out_btns: list[ttk.Button] = []
        self.btns_start: list[ttk.Button] = []
        self.btns_depth: list[ttk.Button] = []

        outer = ttk.Frame(self.root, padding=10)
        outer.grid(sticky="nsew")

        nb = ttk.Notebook(outer)
        nb.grid(row=0, column=0, sticky="nsew")
        self.tab_infer = ttk.Frame(nb, padding=8)
        self.tab_depth = ttk.Frame(nb, padding=8)
        nb.add(self.tab_infer, text=tr("tab.infer"))
        nb.add(self.tab_depth, text=tr("tab.depth"))
        self._build_infer_tab(self.tab_infer)
        self._build_depth_tab(self.tab_depth)

        lf = ttk.LabelFrame(outer, text=tr("log.title"), padding=4)
        lf.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.txt = tk.Text(
            lf, height=12, width=96, state="disabled", wrap="word", **self._font(9)
        )
        sb = ttk.Scrollbar(lf, command=self.txt.yview)
        self.txt.configure(yscrollcommand=sb.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._update_precision()
        self._update_vram()
        self._enable_dnd()

    def _mono_row(self, parent, r: int) -> int:
        """SBS 输入下方的选项: 勾选后点“开始导出”会顺带裁出左半幅单目 8bit 视频 (播放用彩色流)"""
        cf = ttk.Frame(parent)
        cf.grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 6))
        ttk.Checkbutton(
            cf,
            text=tr("mono.check"),
            variable=self.v_mono_export,
        ).pack(side="left")
        return r + 1

    def _build_infer_tab(self, parent):
        r = 0
        ttk.Label(parent, text=tr("sbs.infer.label")).grid(
            row=r, column=0, sticky="w", **PAD
        )
        e = ttk.Entry(parent, width=52, textvariable=self.v_video)
        e.grid(row=r, column=1, **PAD)
        self.video_entries.append((e, True))
        ttk.Button(parent, text=tr("browse"), command=self._pick_video).grid(
            row=r, column=2, **PAD
        )
        r += 1

        r = self._mono_row(parent, r)

        ttk.Checkbutton(
            parent,
            text=tr("both_eyes"),
            variable=self.v_both_eyes,
        ).grid(row=r, column=0, columnspan=3, sticky="w", padx=8)
        r += 1

        mf = ttk.Frame(parent)
        mf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(mf, text=tr("res.label")).pack(side="left")
        cb_size = ttk.Combobox(
            mf,
            width=7,
            textvariable=self.v_depth_size,
            values=("384", "512", "640", "768", "896", "1024"),
        )
        cb_size.pack(side="left", padx=4)
        ttk.Label(mf, text=tr("res.hint")).pack(side="left")
        cb_size.bind("<KeyRelease>", lambda _e: self._update_vram())
        cb_size.bind("<<ComboboxSelected>>", lambda _e: self._update_vram())
        r += 1

        wf = ttk.Frame(parent)
        wf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(wf, text=tr("win.label")).pack(side="left")
        cb_win = ttk.Combobox(
            wf,
            width=7,
            textvariable=self.v_window,
            values=("8", "12", "16", "20", "24", "32"),
        )
        cb_win.pack(side="left", padx=4)
        ttk.Label(wf, text=tr("win.hint")).pack(side="left")
        cb_win.bind("<KeyRelease>", lambda _e: self._update_vram())
        cb_win.bind("<<ComboboxSelected>>", lambda _e: self._update_vram())
        r += 1

        self.lbl_vram = tk.Label(
            parent, text="", justify="left", anchor="w", fg="#8a4b00", **self._font(9)
        )
        self.lbl_vram.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        r += 1

        r = self._common_rows(parent, r)

        r = self._mask_row(parent, r)

        # 辅助功能: 位于“开始导出”上方, 左边缘与上方各项目对齐, 导出到“输出目录”
        df = ttk.Frame(parent)
        df.grid(row=r, column=0, columnspan=3, sticky="w", padx=8, pady=(8, 0))
        btn = ttk.Button(
            df, text=tr("depth_only"), command=self._export_depth_only
        )
        btn.pack(side="left")
        self.btns_depth.append(btn)
        self.btn_meta[btn] = (
            tr("depth_only"),
            self._export_depth_only,
        )
        r += 1

        btn_start = ttk.Button(parent, text=tr("start"), command=self._start_infer)
        btn_start.grid(row=r, column=0, columnspan=3, pady=(6, 4))
        self.btns_start.append(btn_start)
        self.btn_meta[btn_start] = (tr("start"), self._start_infer)

    def _build_depth_tab(self, parent):
        r = 0
        ttk.Label(parent, text=tr("sbs.depth.label")).grid(
            row=r, column=0, sticky="w", **PAD
        )
        e = ttk.Entry(parent, width=52, textvariable=self.v_video)
        e.grid(row=r, column=1, **PAD)
        self.video_entries.append(
            (e, False)
        )  # 该选项卡用已有深度图, 只支持单个视频文件
        ttk.Button(parent, text=tr("browse"), command=self._pick_video).grid(
            row=r, column=2, **PAD
        )
        r += 1

        r = self._mono_row(parent, r)

        ttk.Label(parent, text=tr("depth.dir.label")).grid(
            row=r, column=0, sticky="w", **PAD
        )
        ttk.Entry(parent, width=52, textvariable=self.v_depthfrom).grid(
            row=r, column=1, **PAD
        )
        ttk.Button(parent, text=tr("browse"), command=self._pick_depthdir).grid(
            row=r, column=2, **PAD
        )
        r += 1

        tk.Label(
            parent,
            text=tr("depth.dir.hint"),
            anchor="w",
            fg="#606060",
            **self._font(9),
        ).grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        r += 1

        r = self._common_rows(parent, r)

        r = self._mask_row(parent, r)

        btn_start = ttk.Button(parent, text=tr("start"), command=self._start_from_depth)
        btn_start.grid(row=r, column=0, columnspan=3, pady=(8, 4))
        self.btns_start.append(btn_start)
        self.btn_meta[btn_start] = (tr("start"), self._start_from_depth)

    def _common_rows(self, parent, r: int) -> int:
        """两个选项卡共有的选项: 起止、near/far、编码模式、位深、深度分辨率、输出目录"""
        sf = ttk.Frame(parent)
        sf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(sf, text=tr("range.start")).pack(side="left")
        ttk.Entry(sf, width=10, textvariable=self.v_start).pack(
            side="left", padx=(4, 12)
        )
        ttk.Label(sf, text=tr("range.end")).pack(side="left")
        ttk.Entry(sf, width=10, textvariable=self.v_end).pack(side="left", padx=4)
        ttk.Label(
            sf,
            text=tr("range.hint"),
            foreground="#606060",
        ).pack(side="left")
        r += 1

        tf = ttk.Frame(parent)
        tf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(tf, text=tr("near.label")).pack(side="left")
        e_near = ttk.Entry(tf, width=8, textvariable=self.v_near)
        e_near.pack(side="left", padx=(4, 12))
        ttk.Label(tf, text=tr("far.label")).pack(side="left")
        e_far = ttk.Entry(tf, width=8, textvariable=self.v_far)
        e_far.pack(side="left", padx=4)
        e_near.bind("<KeyRelease>", lambda _e: self._update_precision())
        e_far.bind("<KeyRelease>", lambda _e: self._update_precision())
        r += 1

        mf = ttk.Frame(parent)
        mf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        for text, val in (
            (tr("mode.log"), "log"),
            (tr("mode.linear"), "linear"),
        ):
            ttk.Radiobutton(
                mf,
                text=text,
                value=val,
                variable=self.mode,
                command=self._update_precision,
            ).pack(side="left", padx=(0, 16))
        r += 1

        bf = ttk.Frame(parent)
        bf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(bf, text=tr("bits.label")).pack(side="left")
        for text, val in (
            (tr("bits.8"), 8),
            (tr("bits.10"), 10),
            (tr("bits.12"), 12),
        ):
            ttk.Radiobutton(
                bf,
                text=text,
                value=val,
                variable=self.bits,
                command=self._update_precision,
            ).pack(side="left", padx=(0, 16))
        r += 1

        wf = ttk.Frame(parent)
        wf.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        ttk.Label(
            wf, text=tr("res.note"), foreground="#666"
        ).pack(side="left")
        r += 1

        lbl = tk.Label(
            parent, text="", justify="left", anchor="w", fg="#205080", **self._font(9)
        )
        lbl.grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        self.prec_labels.append(lbl)
        r += 1

        ttk.Label(parent, text=tr("out.label")).grid(row=r, column=0, sticky="w", **PAD)
        e_out = ttk.Entry(parent, width=52, textvariable=self.v_out)
        e_out.grid(row=r, column=1, **PAD)
        b_out = ttk.Button(parent, text=tr("browse"), command=self._pick_outdir)
        b_out.grid(row=r, column=2, **PAD)
        self.out_entries.append(e_out)
        self.out_btns.append(b_out)
        r += 1

        ttk.Checkbutton(
            parent,
            text=tr("same_dir"),
            variable=self.same_dir,
            command=self._update_outstate,
        ).grid(row=r, column=0, columnspan=3, sticky="w", **PAD)
        r += 1
        return r

    def _mask_row(self, parent, r: int) -> int:
        """可选 mask: 一个浏览按钮 + PIL 预览缩略图 + 提示文字"""
        lf = ttk.LabelFrame(
            parent,
            text=tr("mask.frame"),
            padding=4,
        )
        lf.grid(row=r, column=0, columnspan=3, sticky="we", padx=8, pady=(8, 4))
        row = ttk.Frame(lf)
        row.grid(row=0, column=0, sticky="w")
        ttk.Label(row, text=tr("mask.label"), width=12).pack(side="left")
        e = ttk.Entry(row, width=48, textvariable=self.v_mask)
        e.pack(side="left", padx=4)
        ttk.Button(row, text=tr("browse"), command=self._pick_mask).pack(side="left")
        prev = tk.Label(
            row,
            width=12,
            height=5,
            bg="#202020",
            relief="solid",
            borderwidth=1,
            text=tr("mask.preview"),
        )
        prev.pack(side="left", padx=(8, 0))
        self._mask_prevs.append(prev)

        # 路径自动补全: 如果视频同目录存在 mask.jpg, 自动填入 (预览由 trace 自动刷新)
        self._autofill_mask_from_video()
        self._update_mask_preview()

        if not _HAS_PIL:
            tk.Label(
                parent,
                text=tr("mask.pil_hint"),
                anchor="w",
                fg="#808080",
                **self._font(8),
            ).grid(row=r + 1, column=0, columnspan=3, sticky="w", padx=8)
        return r + 1

    def _autofill_mask_from_video(self):
        """视频同目录存在 mask.jpg 时自动填入 (方便重复使用)"""
        v = self.v_video.get().strip()
        if not v or self.v_mask.get().strip():
            return
        p = Path(v)
        base = p if p.is_dir() else p.parent
        for name in ("mask.jpg", "mask.png", "mask1.jpg"):  # 兼容之前命名
            auto = base / name
            if auto.is_file():
                self.v_mask.set(str(auto))  # trace 会自动刷新预览
                break

    def _pick_mask(self):
        p = filedialog.askopenfilename(
            title=tr("mask.pick_title"),
            filetypes=[
                (tr("filter.images"), "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                (tr("filter.all"), "*.*"),
            ],
        )
        if p:
            self.v_mask.set(p)  # trace 会自动刷新预览

    def _schedule_mask_preview(self):
        """路径变化时防抖刷新 (手输时避免每敲一个字符就开一次文件)"""
        if self._mask_prev_job is not None:
            try:
                self.root.after_cancel(self._mask_prev_job)
            except Exception:
                pass
        self._mask_prev_job = self.root.after(200, self._update_mask_preview)

    def _update_mask_preview(self):
        """用 PIL 加载 mask, 缩放到 MASK_PREVIEW_SIZE 保持比例直接显示 (不裁切、不二值化)"""
        self._mask_prev_job = None
        if not _HAS_PIL or not self._mask_prevs:
            return
        path = self.v_mask.get().strip()
        if not path or not Path(path).is_file():
            self._mask_img_ref = None
            for lb in self._mask_prevs:
                lb.config(
                    image="", text=tr("mask.preview"), width=12, height=5, bg="#202020"
                )
            return
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(MASK_PREVIEW_SIZE, Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self._mask_img_ref = photo
            for lb in self._mask_prevs:
                lb.config(
                    image=photo,
                    text="",
                    width=photo.width(),
                    height=photo.height(),
                    bg="#404040",
                )
        except Exception as e:
            self._log(tr("log.mask_preview_fail", e=e))
            self._mask_img_ref = None
            for lb in self._mask_prevs:
                lb.config(image="", text=tr("mask.load_fail"), bg="#802020")

    # ---------- 浏览按钮 ----------

    def _enable_dnd(self):
        """SBS 视频输入框接受文件拖放 (依赖 tkinterdnd2; 没装则静默跳过, 仍可用“浏览…”选文件)"""
        if TkinterDnD is None:
            return
        for w, allow_dir in self.video_entries:
            try:
                w.drop_target_register(DND_FILES)
                w.dnd_bind("<<Drop>>", lambda ev, ok=allow_dir: self._on_drop(ev, ok))
            except tk.TclError:
                # tkdnd 扩展未随根窗口加载: 放弃拖放, 仍可用“浏览…”选文件
                return

    def _videos(self, allow_dir: bool = True) -> list[Path]:
        """输入框里的路径: 视频文件 -> [自己]; 文件夹 -> 目录下所有视频 (跳过已导出的单目视频)"""
        p = Path(self.v_video.get().strip())
        if p.is_file():
            return [p] if p.suffix.lower() in VIDEO_EXT else []
        if p.is_dir() and allow_dir:
            return sorted(
                f
                for f in p.iterdir()
                if f.is_file()
                and f.suffix.lower() in VIDEO_EXT
                and not f.stem.endswith(MONO_SUFFIX)
            )
        return []

    def _outdir_for(self, video: Path) -> Path:
        """勾选“视频同目录”时每个视频各自输出到自己所在目录"""
        out = video.parent if self.same_dir.get() else Path(self.v_out.get().strip())
        out.mkdir(parents=True, exist_ok=True)
        return out

    def _next_stem(self, outdir: Path, stem: str, mono: bool, depth_npy: bool) -> str:
        """返回不与已有导出文件冲突的 stem: <stem>, <stem>_1, <stem>_2, ...

        检查的产物 (由 process_video.py 与 mono 转码生成):
          - 视频导出: <stem>_depth.mp4, <stem>.json, (勾选单目时) <stem>_mono8.mp4
          - 深度缓存: <stem>_depth_npy/ 目录
        """
        for s in (stem, *(f"{stem}_{i}" for i in range(1, 1000))):
            if depth_npy:
                if not (outdir / f"{s}_depth_npy").exists():
                    return s
            else:
                conflicts = [outdir / f"{s}_depth.mp4", outdir / f"{s}.json"]
                if mono:
                    conflicts.append(outdir / f"{s}_mono8.mp4")
                if not any(c.exists() for c in conflicts):
                    return s
        return stem  # 兜底 (理论上不会到 1000)

    def _on_drop(self, event, allow_dir: bool = True):
        paths = []
        for p in self.root.tk.splitlist(event.data):
            p = Path(p.strip().strip('"'))
            if p.is_dir():  # 拖文件夹: 直接用它, 导出时处理目录下所有视频
                if not allow_dir:
                    self._log(tr("log.drop_ignored", p=p))
                    continue
                paths.append(p)
                break
            if p.suffix.lower() in VIDEO_EXT:
                paths.append(p)
        if not paths:
            return getattr(event, "action", None)
        target = paths[0]
        self.v_video.set(str(target))  # 两个选项卡共享同一个变量, 拖一次两处同步
        n = len(self._videos(allow_dir))
        extra = tr("log.drop_dir_extra", n=n) if target.is_dir() else ""
        self._log(tr("log.drop", p=target, extra=extra))
        return getattr(event, "action", None)

    def _pick_video(self):
        p = filedialog.askopenfilename(
            title=tr("dlg.video"),
            filetypes=[
                (tr("filter.videos"), "*.mp4 *.mkv *.mov *.avi"),
                (tr("filter.all"), "*.*"),
            ],
        )
        if p:
            self.v_video.set(p)

    def _pick_depthdir(self):
        p = filedialog.askdirectory(title=tr("dlg.depthdir"))
        if p:
            self.v_depthfrom.set(p)

    def _pick_outdir(self):
        p = filedialog.askdirectory(title=tr("dlg.outdir"))
        if p:
            self.v_out.set(p)

    def _mono_out_path(self, video: Path, stem: str = "") -> Path:
        outdir = (
            video.parent
            if self.same_dir.get()
            else Path(self.v_out.get().strip() or ".")
        )
        s = stem or video.stem
        return outdir / (s + MONO_SUFFIX + ".mp4")

    def _mono_cmd(
        self, video: str, out: Path, start: float = 0.0, end: float = -1.0
    ) -> list[str]:
        """SBS -> 左半幅单目 8bit HEVC: 播放端只需解 4K 而非 8K SBS, 卡顿明显减轻

        start >= 0 时从该秒数开始; end > start 时在该秒数停止 (end=-1 表示到末尾)。
        """
        if has_nvenc():
            # cq 21 + 40M 上限: 实测 4096x4096 8bit 约 50 Mbps, 8K 源 146 Mbps 的一半以内
            enc = [
                "-c:v",
                "hevc_nvenc",
                "-preset",
                "p5",
                "-tune",
                "hq",
                "-rc",
                "vbr",
                "-cq",
                "21",
                "-b:v",
                "40M",
                "-maxrate",
                "50M",
                "-bufsize",
                "100M",
            ]
        else:
            enc = ["-c:v", "libx265", "-preset", "faster", "-crf", "20"]
        # 输入前 -ss 快速 seek (ffmpeg 会解码到精确位置); 用 -t 时长而非 -to:
        # -to 在 -ss 前置时会以“seek 后的 0 点”为基准, 导致导出长度错误
        pre = ["-ss", f"{start:g}"] if start > 0 else []
        post_t = ["-t", f"{end - start:g}"] if end > start else []
        return (
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-stats",
                "-stats_period",
                "2",
            ]
            + pre
            + ["-i", video]
            + post_t
            + [
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-dn",
                "-sn",  # 视频轨 + 音轨(有则带, 没有源也能跑); 丢 tmcd 附加流
                "-vf",
                "crop=iw/2:ih:0:0,setpts=PTS-STARTPTS",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
            ]
            + enc
            + [
                "-pix_fmt",
                "yuv420p",
                "-tag:v",
                "hvc1",
                "-movflags",
                "+faststart",
                "-write_tmcd",
                "0",
                str(out),
            ]
        )

    def _update_outstate(self):
        """勾选“视频同目录”时置灰输出目录输入"""
        state = "disabled" if self.same_dir.get() else "normal"
        for e in self.out_entries:
            e.config(state=state)
        for b in self.out_btns:
            b.config(state=state)

    # ---------- 精度 / 显存提示 ----------

    def _near_far(self) -> tuple[float, float]:
        try:
            return float(self.v_near.get()), float(self.v_far.get())
        except ValueError:
            return 0.0, -1.0

    def _update_precision(self):
        near, far = self._near_far()
        text = precision_text(near, far, self.mode.get(), self.bits.get())
        for lbl in self.prec_labels:
            lbl.config(text=text)

    def _update_vram(self):
        try:
            w = int(self.v_depth_size.get().strip())
            win = int(self.v_window.get().strip())
            if w < 32 or win < 4:
                raise ValueError
        except ValueError:
            self.lbl_vram.config(text=tr("vram.bad"))
            return
        gb = vram_estimate(w, win)
        self.lbl_vram.config(text=tr("vram.est", gb=gb))

    # ---------- 导出 ----------

    def _log(self, s: str):
        self.log_q.put(s)

    def _poll_log(self):
        while not self.log_q.empty():
            line = self.log_q.get_nowait()
            self.txt.config(state="normal")
            self.txt.insert("end", line + "\n")
            self.txt.see("end")
            self.txt.config(state="disabled")
        self.root.after(80, self._poll_log)

    @staticmethod
    def _parse_time(s: str) -> float:
        """解析时间字符串为秒数。支持:
        - 纯数字       -> 直接按秒 (如 12.5)
        - "aa:bb"      -> aa 分钟, bb 秒 (如 1:30 = 90)
        - "aa:bb:cc"   -> aa 小时, bb 分钟, cc 秒 (如 1:02:03 = 3723)
        解析失败或出现负数抛 ValueError
        """
        s = s.strip()
        if not s:
            raise ValueError("空字符串")
        if ":" in s:
            parts = s.split(":")
            if len(parts) == 2:
                vals = [float(p) for p in parts]
                m, sec = vals
                if m < 0 or sec < 0:
                    raise ValueError(f"时间分量不能为负: {s!r}")
                return m * 60 + sec
            elif len(parts) == 3:
                vals = [float(p) for p in parts]
                h, m, sec = vals
                if h < 0 or m < 0 or sec < 0:
                    raise ValueError(f"时间分量不能为负: {s!r}")
                return h * 3600 + m * 60 + sec
            else:
                raise ValueError(f"时间格式错误: {s!r} (最多支持 HH:MM:SS)")
        sec = float(s)
        if sec < 0:
            raise ValueError(f"时间不能为负: {s!r}")
        return sec

    def _collect_params(self, allow_dir: bool = True) -> dict | None:
        """收集并校验通用参数, 返回参数字典; 校验失败弹框并返回 None"""
        videos = self._videos(allow_dir)
        if not videos:
            messagebox.showerror(
                tr("err.title"),
                tr("err.input") if allow_dir else tr("err.input_file"),
            )
            return None
        near, far = self._near_far()
        if not (0 < near < far):
            messagebox.showerror(tr("err.title"), tr("err.near_far"))
            return None
        try:
            start = self._parse_time(self.v_start.get() or "0")
            end_str = self.v_end.get().strip()
            end = self._parse_time(end_str) if end_str else -1.0
            if start < 0 or (end >= 0 and end <= start):
                raise ValueError("range")
        except Exception:
            messagebox.showerror(tr("err.title"), tr("err.time"))
            return None
        # 勾选“视频同目录”时每个视频导出到自己所在目录 (在构建任务时逐视频计算)
        if not self.same_dir.get():
            outdir = Path(self.v_out.get().strip())
            if not outdir.is_dir() and not outdir.parent.is_dir():
                messagebox.showerror(tr("err.title"), tr("err.out"))
                return None
            outdir.mkdir(parents=True, exist_ok=True)
        return {"videos": videos, "near": near, "far": far, "start": start, "end": end}

    def _infer_params(self) -> tuple[int, int] | None:
        try:
            depth_size = int(self.v_depth_size.get().strip())
            window = int(self.v_window.get().strip())
            if depth_size < 32 or window < 4:
                raise ValueError
        except ValueError:
            messagebox.showerror(tr("err.title"), tr("err.res_window"))
            return None
        return depth_size, window

    def _base_cmd(self, p: dict, video: Path) -> list[str]:
        cmd = [
            sys.executable,
            str(PROCESS),
            "--input-video",
            str(video),
            "--start",
            f"{p['start']:g}",
            "--end",
            f"{p['end']:g}",
            "--min-range",
            f"{p['near']:g}",
            "--max-range",
            f"{p['far']:g}",
            "--depth-encode",
            self.mode.get(),
            "--depth-bits",
            str(self.bits.get()),
        ]
        m = self.v_mask.get().strip()
        if m:
            cmd += ["--mask", m]
        return cmd

    def _mono_step(
        self, p: dict, video: Path, cmd: list[str], stem: str = ""
    ) -> list[tuple[list[str], Path]]:
        """勾选“导出单目 8bit 视频”时: 先把 SBS 裁成左半幅 (返回待执行命令), 并让 json 指向它"""
        if not self.v_mono_export.get():
            return []
        mono = self._mono_out_path(video, stem)
        mono.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--color-file", str(mono), "--color-layout", "mono"]
        return [(self._mono_cmd(str(video), mono, p["start"], p["end"]), mono)]

    def _start_infer(self):
        """选项卡1: 视频推理 -> 深度视频 mp4 + json (输入框是文件夹时逐个视频导出)"""
        if self.running:
            return
        p = self._collect_params()
        if p is None:
            return
        infer = self._infer_params()
        if infer is None:
            return
        depth_size, window = infer
        jobs = []
        mono_on = self.v_mono_export.get()
        for v in p["videos"]:
            outdir = self._outdir_for(v)
            stem = self._next_stem(outdir, v.stem, mono=mono_on, depth_npy=False)
            outmp4 = outdir / (stem + ".mp4")
            cmd = self._base_cmd(p, v) + [
                "--pack",
                "gray10mp4",
                "--depth-size",
                str(depth_size),
                "--window",
                str(window),
                "--encoder",
                "auto",
                "--out",
                str(outmp4),
            ]
            if self.v_both_eyes.get():
                cmd.append("--both-eyes")
            jobs.append((cmd, "video", outmp4, self._mono_step(p, v, cmd, stem)))
        self._run(jobs)

    def _start_from_depth(self):
        """选项卡2: 已有深度图 -> 深度视频 mp4 + json (只支持单个视频文件)"""
        if self.running:
            return
        p = self._collect_params(allow_dir=False)
        if p is None:
            return
        depth_dir = self.v_depthfrom.get().strip()
        if not depth_dir or not Path(depth_dir).is_dir():
            messagebox.showerror(tr("err.title"), tr("err.depthdir"))
            return
        v = p["videos"][0]
        outdir = self._outdir_for(v)
        stem = self._next_stem(
            outdir, v.stem, mono=self.v_mono_export.get(), depth_npy=False
        )
        outmp4 = outdir / (stem + ".mp4")
        cmd = self._base_cmd(p, v) + [
            "--pack",
            "gray10mp4",
            "--encoder",
            "auto",
            "--out",
            str(outmp4),
            "--depth-from",
            depth_dir,
        ]
        self._run([(cmd, "video", outmp4, self._mono_step(p, v, cmd, stem))])

    def _export_depth_only(self):
        """选项卡1: 直接推理, float32 原始毫米 .npy 导出到 输出目录/<名字>_depth_npy"""
        if self.running:
            return
        p = self._collect_params()
        if p is None:
            return
        infer = self._infer_params()
        if infer is None:
            return
        depth_size, window = infer
        jobs = []
        for v in p["videos"]:
            outdir = self._outdir_for(v)
            stem = self._next_stem(outdir, v.stem, mono=False, depth_npy=True)
            depth_dir = outdir / (stem + "_depth_npy")
            depth_dir.mkdir(parents=True, exist_ok=True)
            cmd = self._base_cmd(p, v) + [
                "--pack",
                "depthnpy",
                "--depth-size",
                str(depth_size),
                "--window",
                str(window),
                "--depth-cache",
                str(depth_dir),
            ]
            if self.v_both_eyes.get():
                # 双目: npy 分别写到 <dir>\L 与 <dir>\R
                cmd.append("--both-eyes")
            jobs.append((cmd, "depth_npy", depth_dir, []))
        self._run(jobs)

    def _run(self, jobs: list):
        self.running = True
        self._cancel.clear()
        for b in self.btns_start + self.btns_depth:
            b.config(state="normal", text=tr("cancel"), command=self._cancel_run)
        n = len(jobs)
        if n > 1:
            self._log(tr("log.batch_n", n=n))
        for i, (cmd, _kind, _target, mono) in enumerate(jobs, 1):
            prefix = f"[{i}/{n}] " if n > 1 else ""
            for mc, _mout in mono:
                self._log(prefix + f"$ {' '.join(mc)}")
            self._log(prefix + f"$ {' '.join(cmd)}")
        threading.Thread(target=self._worker, args=(jobs,), daemon=True).start()

    def _cancel_run(self):
        """导出中按钮的功能: 置取消标记并杀掉当前子进程 (taskkill /T 连带 ffmpeg)"""
        if not self.running or self._cancel.is_set():
            return
        self._cancel.set()
        self._log(tr("log.cancel_stop"))
        self._kill_proc()

    def _kill_proc(self):
        p = self._proc
        if p is None or p.poll() is not None:
            return
        try:
            if os.name == "nt":
                # /T 杀整棵进程树: process_video.py 自己还会再拉起 ffmpeg
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(p.pid)],
                    capture_output=True,
                    creationflags=NO_WINDOW,
                )
            else:
                p.terminate()
        except Exception as e:
            self._log(tr("log.cancel_kill", e=e))

    def _exec(self, cmd: list[str]) -> int:
        """跑一个子进程并把输出逐行写入日志 (进度行以 \\r 结尾)"""
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # 否则中文在 GBK 控制台码页下会乱码
        env["PYTHONUNBUFFERED"] = "1"
        p = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=NO_WINDOW,
        )
        self._proc = p
        buf = ""
        try:
            while True:
                ch = p.stdout.read(1)
                if not ch:
                    break
                if ch in "\r\n":
                    if buf:
                        self._log(buf)
                    buf = ""
                else:
                    buf += ch
            if buf:
                self._log(buf)
            return p.wait()
        finally:
            self._proc = None

    def _worker(self, jobs: list):
        """子线程: 依次处理每个视频 (单个失败不中断后续), 结束时汇总"""
        n = len(jobs)
        ok = 0
        try:
            for i, (cmd, kind, target, mono) in enumerate(jobs, 1):
                if self._cancel.is_set():
                    break
                head = f"({i}/{n}) " if n > 1 else ""
                if self._job(cmd, kind, target, mono, head):
                    ok += 1
        except Exception as e:
            self._log(tr("log.exception", e=e))
        finally:
            if self._cancel.is_set():
                self._log(tr("log.cancel_done"))
            elif n > 1:
                self._log(tr("log.batch_end", ok=ok, n=n))
            self.root.after(0, self._done)

    def _job(
        self, cmd: list[str], kind: str, target: Path, mono: list, head: str
    ) -> bool:
        """单个视频: 先导单目 8bit 彩色视频 (若勾选), 再导深度; 返回是否全部成功"""
        try:
            for mc, mout in mono:
                if self._cancel.is_set():
                    return False
                if mout.exists():
                    self._log(head + tr("log.skip_mono", p=mout))
                else:
                    self._log(head + tr("log.mono", name=mout.name))
                    code = self._exec(mc)
                    if self._cancel.is_set():
                        return False
                    if code != 0:
                        self._log(head + tr("log.mono_fail", code=code))
                        return False
                    self._log(
                        head
                        + tr(
                            "log.mono_done",
                            p=mout,
                            mb=mout.stat().st_size / 1e6,
                        )
                    )
                # json 已存在时先就地指向单目视频; 下面 process_video 也会重写一遍
                if patch_color(target.with_suffix(".json"), mout.name):
                    self._log(
                        head
                        + "    "
                        + tr("log.json_patched", stem=target.stem, name=mout.name)
                    )

            code = self._exec(cmd)
            if self._cancel.is_set():
                return False
            if code == 0:
                if kind == "depth_npy":
                    npys = sorted(target.glob("*.npy"))
                    msg = head + tr("log.npy_done", p=target, n=len(npys))
                    meta = target / "meta.json"
                    if meta.exists():
                        msg += f"\n    {meta.name}"
                    self._log(msg)
                else:
                    depth = target.with_name(target.stem + "_depth.mp4")
                    meta = target.with_suffix(".json")
                    msg = head + tr("log.video_done")
                    for f in (depth, meta):
                        if f.exists():
                            msg += f"\n    {f.name}  ({f.stat().st_size / 1e6:.1f} MB)"
                    self._log(msg)
                return True
            self._log(head + tr("log.depth_fail", code=code))
            return False
        except Exception as e:
            self._log(head + tr("log.exception", e=e))
            return False

    def _done(self):
        self.running = False
        self._proc = None
        for b, (text, cmd) in self.btn_meta.items():
            b.config(state="normal", text=text, command=cmd)


if __name__ == "__main__":
    # 有 tkinterdnd2 就用它创建根窗口 (才能接收文件拖放), 否则退回普通 Tk
    root = TkinterDnD.Tk() if TkinterDnD is not None else tk.Tk()
    App(root)
    root.mainloop()
