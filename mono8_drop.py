# -*- coding: utf-8 -*-
"""单目 8bit HEVC 导出 (命令行 / 拖放版)

用法: 把 SBS 视频文件拖到本文件图标上, 或  python mono8_drop.py 视频路径 [更多...]
  - 取左半幅 (左眼), 转 8bit HEVC, 输出到视频所在文件夹: <原名>_mono8.mp4
  - 结束时 pause, 窗口不会一闪而过

有 N 卡走 hevc_nvenc (cq 21, 上限 50M), 否则退回 libx265 crf 20.
"""

import subprocess
import sys
from pathlib import Path

SUFFIX = "_mono8"
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".m4v", ".webm"}


def has_nvenc() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
        return "hevc_nvenc" in out
    except Exception:
        return False


def build_cmd(video: Path, out: Path) -> list:
    if has_nvenc():
        enc = ["-c:v", "hevc_nvenc", "-preset", "p5", "-tune", "hq",
               "-rc", "vbr", "-cq", "21", "-b:v", "40M", "-maxrate", "50M", "-bufsize", "100M"]
    else:
        enc = ["-c:v", "libx265", "-preset", "faster", "-crf", "20"]
    return ["ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-stats", "-stats_period", "2",
            "-i", str(video),
            "-map", "0:v:0", "-map", "0:a:0?", "-dn", "-sn",     # 视频轨 + 音轨(有则带, 没有源也能跑); 丢 tmcd 附加流
            "-vf", "crop=iw/2:ih:0:0",                          # 左半幅 (左眼), 右半丢弃
            "-c:a", "aac", "-b:a", "128k"] + enc + [             # 音频: AAC 128k, 源无音轨时 -map 后面的 ? 自动跳过
            "-pix_fmt", "yuv420p", "-tag:v", "hvc1", "-movflags", "+faststart",
            "-write_tmcd", "0",                                  # 不带 tmcd 时间码轨
            str(out)]


def export(video: Path) -> bool:
    out = video.with_name(video.stem + SUFFIX + ".mp4")
    print(f"\n[导出] {video.name}\n       -> {out.name}")
    code = subprocess.run(build_cmd(video, out)).returncode   # 直接继承控制台, 进度实时可见
    if code != 0:
        print(f"[失败] {video.name}: ffmpeg 退出码 {code}")
        return False
    print(f"[完成] {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    args = [Path(a.strip('"')) for a in sys.argv[1:] if Path(a.strip('"')).is_file()]
    files = [p for p in args if p.suffix.lower() in VIDEO_EXT]
    if not files:
        line = input("请输入 SBS 视频路径 (多个用空格分隔, 直接把文件拖进来也行): ").strip()
        files = [Path(p.strip('"').strip()) for p in line.split()]
    files = [p for p in files if p.is_file() and p.suffix.lower() in VIDEO_EXT]
    if not files:
        print("[错误] 没有可用的视频文件")
    else:
        ok = sum(export(p) for p in files)
        print(f"\n结束: 成功 {ok} / 共 {len(files)}")
    try:
        input("\n按回车键退出…")
    except EOFError:       # 被管道/重定向调用时没有 stdin, 直接退出
        pass


if __name__ == "__main__":
    main()
