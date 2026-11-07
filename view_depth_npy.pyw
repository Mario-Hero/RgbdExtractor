"""深度 .npy 查看器 (tkinter)

读取 process_video.py 导出的深度缓存: 单帧 float32 .npy, 单位毫米, 未钳制 (0 = 无效)。
可实时调整 near / far 与编码方式 (log / linear), 观察深度图在不同量化参数下的效果。

  · 打开文件: 选单个 .npy
  · 打开目录: 选一个 *_depth_npy 目录, 逐帧浏览 (← → 切换帧)
  · 位深 8/10/12: 与导出时的 --depth-bits 对应, 只影响码值量化, 显示统一映射到 8bit

用法: python view_depth_npy.pyw [某个 .npy 或目录]
"""

import sys
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np


def encode_depth_codes(depth_m: np.ndarray, near: float, far: float,
                       encode: str, bits: int) -> np.ndarray:
    """米制深度 -> N bit 码值 (0..max_code), 与 process_video.encode_depth_gray10 一致

    这里独立实现而不再 import process_video: 后者会连带加载 torch/CUDA, 拖慢启动。
    """
    max_code = (1 << bits) - 1
    d = np.clip(depth_m.astype(np.float64), near, far)
    if encode == "log":
        v = np.log(d / near) / np.log(far / near)
    else:
        v = (d - near) / (far - near)
    v = np.clip(v, 0.0, 1.0)
    v[depth_m <= 0] = 1.0          # 无效深度按最远端处理 (与导出端一致)
    return np.round(v * max_code).astype(np.uint16)


class DepthViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("深度 .npy 查看器")
        root.geometry("980x820")

        self.files: list[Path] = []
        self.idx = 0
        self.raw: np.ndarray | None = None

        self._tmpdir = Path(tempfile.mkdtemp(prefix="depth_view_"))
        self._tmpseq = 0
        self._tmp_last: Path | None = None
        self._photo: tk.PhotoImage | None = None
        self._job = None

        self.v_path = tk.StringVar()
        self.v_near = tk.DoubleVar(value=0.05)
        self.v_far = tk.DoubleVar(value=5.0)
        self.v_enc = tk.StringVar(value="log")
        self.v_bits = tk.StringVar(value="10")
        self.v_frame = tk.StringVar(value="未加载")
        self.v_info = tk.StringVar(value="")

        self._build()

    # ---------------- UI ----------------
    def _build(self):
        pad = {"padx": 4, "pady": 3}
        top = ttk.Frame(self.root, padding=8)
        top.pack(fill="x")

        ttk.Entry(top, textvariable=self.v_path).grid(row=0, column=0, columnspan=3,
                                                     sticky="ew", **pad)
        ttk.Button(top, text="打开文件", command=self.open_file).grid(row=0, column=3, **pad)
        ttk.Button(top, text="打开目录", command=self.open_dir).grid(row=0, column=4, **pad)
        top.columnconfigure(0, weight=1)

        nav = ttk.Frame(top)
        nav.grid(row=1, column=0, columnspan=5, sticky="w", **pad)
        ttk.Button(nav, text="◀ 上一帧", command=lambda: self.step(-1)).pack(side="left")
        ttk.Button(nav, text="下一帧 ▶", command=lambda: self.step(1)).pack(side="left", padx=6)
        ttk.Label(nav, textvariable=self.v_frame).pack(side="left", padx=8)

        rng = ttk.Frame(top)
        rng.grid(row=2, column=0, columnspan=5, sticky="ew", **pad)

        ttk.Label(rng, text="near(m)").grid(row=0, column=0, sticky="w")
        ttk.Entry(rng, textvariable=self.v_near, width=8).grid(row=0, column=1, **pad)
        ttk.Scale(rng, from_=0.01, to=2.0, variable=self.v_near,
                  command=lambda _: self.schedule()).grid(row=0, column=2, sticky="ew", **pad)

        ttk.Label(rng, text="far(m)").grid(row=0, column=3, sticky="w")
        ttk.Entry(rng, textvariable=self.v_far, width=8).grid(row=0, column=4, **pad)
        ttk.Scale(rng, from_=1.0, to=100.0, variable=self.v_far,
                  command=lambda _: self.schedule()).grid(row=0, column=5, sticky="ew", **pad)

        for col in (2, 5):
            rng.columnconfigure(col, weight=1)

        opt = ttk.Frame(top)
        opt.grid(row=3, column=0, columnspan=5, sticky="w", **pad)
        ttk.Label(opt, text="编码:").pack(side="left")
        for text, val in (("log (相对误差恒定)", "log"), ("linear (绝对误差恒定)", "linear")):
            ttk.Radiobutton(opt, text=text, value=val, variable=self.v_enc,
                            command=self.schedule).pack(side="left", padx=(6, 0))
        ttk.Label(opt, text="   位深:").pack(side="left")
        cb = ttk.Combobox(opt, textvariable=self.v_bits, values=["8", "10", "12"],
                          width=4, state="readonly")
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _: self.schedule())

        ttk.Label(top, textvariable=self.v_info, foreground="#444").grid(
            row=4, column=0, columnspan=5, sticky="w", **pad)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.schedule(120))

        self.root.bind("<Left>", lambda _: self.step(-1))
        self.root.bind("<Right>", lambda _: self.step(1))

        # 输入框回车/失焦时应用数值
        for w in top.winfo_children():
            self._bind_commit(w)

    def _bind_commit(self, widget):
        for child in widget.winfo_children():
            if isinstance(child, ttk.Entry):
                child.bind("<Return>", lambda _: self.schedule(0))
                child.bind("<FocusOut>", lambda _: self.schedule(60))
            self._bind_commit(child)

    # ---------------- 加载 ----------------
    def open_file(self):
        p = filedialog.askopenfilename(title="选择深度 .npy",
                                       filetypes=[("深度 npy", "*.npy"), ("所有文件", "*.*")])
        if p:
            self.load_paths([Path(p)])

    def open_dir(self):
        d = filedialog.askdirectory(title="选择深度缓存目录 (*_depth_npy)")
        if not d:
            return
        files = sorted(Path(d).glob("*.npy"))
        if not files:
            messagebox.showwarning("提示", "该目录下没有 .npy 文件")
            return
        self.load_paths(files)

    def load_paths(self, files: list[Path]):
        self.files = files
        self.idx = 0
        self._load_current()

    def step(self, delta: int):
        if not self.files:
            return
        i = max(0, min(len(self.files) - 1, self.idx + delta))
        if i != self.idx:
            self.idx = i
            self._load_current()

    def _load_current(self):
        p = self.files[self.idx]
        self.v_path.set(str(p))
        self.v_frame.set(f"第 {self.idx + 1} / {len(self.files)} 帧  {p.name}")
        try:
            raw = np.load(p).astype(np.float32)
        except Exception as e:
            messagebox.showerror("读取失败", f"{p}\n{e}")
            return
        if raw.ndim != 2:
            messagebox.showerror("格式错误", f"需要二维数组, 实际 shape={raw.shape}")
            return
        self.raw = raw
        self.render()

    # ---------------- 渲染 ----------------
    def schedule(self, delay: int = 30):
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(int(delay), self.render)

    def _params(self):
        try:
            near = float(self.v_near.get())
            far = float(self.v_far.get())
        except (tk.TclError, ValueError):
            return None
        near = max(near, 1e-4)
        far = max(far, near * 1.001)      # far 必须 > near
        return near, far

    def render(self):
        self._job = None
        if self.raw is None:
            return
        prm = self._params()
        if prm is None:
            return
        near, far = prm
        bits = int(self.v_bits.get())
        enc = self.v_enc.get()

        depth = self.raw / 1000.0          # 原始毫米 -> 米
        code = encode_depth_codes(depth, near, far, enc, bits)
        max_code = (1 << bits) - 1
        img8 = np.round(code.astype(np.float32) * (255.0 / max_code)).astype(np.uint8)

        self._show(img8)
        self._update_info(depth, near, far, bits, enc)

    def _show(self, img8: np.ndarray):
        cw = max(self.canvas.winfo_width(), 64)
        ch = max(self.canvas.winfo_height(), 64)
        h, w = img8.shape
        s = min(cw / w, ch / h)
        tw, th = max(1, int(round(w * s))), max(1, int(round(h * s)))
        view = img8 if (tw, th) == (w, h) else cv2.resize(
            img8, (tw, th), interpolation=cv2.INTER_NEAREST)

        self._tmpseq += 1
        tmp = self._tmpdir / f"v{self._tmpseq % 4}.png"
        cv2.imwrite(str(tmp), view)
        old = self._tmp_last
        self._photo = tk.PhotoImage(file=str(tmp))
        self._tmp_last = tmp
        if old is not None and old != tmp and old.exists():
            try:
                old.unlink()
            except OSError:
                pass

        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo)

    def _update_info(self, depth: np.ndarray, near: float, far: float, bits: int, enc: str):
        valid = depth[depth > 0]
        if valid.size:
            v = (f"有效 {valid.size / depth.size * 100:.1f}%  "
                 f"范围 {valid.min():.2f} / {np.median(valid):.2f} / {np.percentile(valid, 99):.2f} m"
                 f" (最小/中值/99%)")
        else:
            v = "该帧无有效深度"
        self.v_info.set(f"{depth.shape[1]}x{depth.shape[0]}  "
                        f"near={near:g} far={far:g}  {enc} {bits}bit  |  {v}")


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    app = DepthViewer(root)
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.is_dir():
            files = sorted(p.glob("*.npy"))
            if files:
                app.load_paths(files)
        elif p.is_file():
            app.load_paths([p])
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:      # .pyw 无控制台, 弹窗显示异常
        r = tk.Tk()
        r.withdraw()
        messagebox.showerror("启动失败", repr(exc))
        r.destroy()
        raise
