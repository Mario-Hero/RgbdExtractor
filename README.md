# RgbdExtractor

用 Stereo Any Video 获取 VR180 双目画面的深度图，并打包成可在 PC 串流到头显观看的深度视频（mp4 + 同名 json）。
Turn VR180 stereo footage into a depth map with Stereo Any Video, then pack it into a depth-displacement video (mp4 + sibling json) which can be streamed to a head-mounted display from a PC.

视频立体模型按 20 帧滑窗推理，时间一致性由模型提供。
The video stereo model runs inference on 20-frame sliding windows, and temporal consistency comes from the model itself.

**深度视频播放请见 [RgbdPlayer](https://github.com/Mario-Hero/RgbdPlayer)。**
**For playback see [RgbdPlayer](https://github.com/Mario-Hero/RgbdPlayer).**

## 这是一个失败的尝试 This is a failed attempt

在观看VR180 3D视频的时候，相信大家都注意到了，如果头部在roll方向旋转（也就是歪头），近处的画面会发生重叠，导致观感下降。并且VR180是不能前后左右移动的，也就是只有3dof，没有6dof。所以为了进一步提高沉浸感，最近开始流行一种4DGS的视频，可以在空间中自由移动，也可以随意歪头。

但是这种视频需要非常多相机才能拍摄，普通人根本没法拍。如果能把VR180 3D视频转成4DGS那就好了。从双目图像获取深度的研究已经很成熟了（甚至从2d图像获取深度也有了），单目转4dgs的也有，但是双目转4dgs的确实没有。

所以我就退而求其次，用了比较经典的球壳+凹凸的办法播放RGB+深度画面。双目深度获取模型选的2025年的Stereo Any Video，这个时域稳定性比较好。2026年的c-fast-foundationstereo速度挺快的，但是时域不稳定。Stereo Any Video要输入许多帧用于滑动窗口，会导致显存占用过大，目前我用的是640x640分辨率，20帧滑动窗口，12G显存够用。

播放效果如何呢？我只能说稍微有点用，头部可以大幅旋转、轻微移动，并且始终有立体感。如果头移动太远，深度之间的“悬崖”会拉丝得非常严重，影响观感。

深度图的存储，我是用了上下限+对数+10bit HEVC 视频编码，保证近距离精度能到半厘米。但是我感觉模型好像本身也没有这么高精度……

而且这种方式，把双目视频转换成了左边单目+深度，那右边的RGB信息不就丢失了？假如左眼对齐原来的左眼视角，那右眼观看的画面必定会偏移，产生拉丝。

我又尝试左右颠倒，再获取右眼的深度，合并成一个视频。右眼模糊的问题是解决了，但是两个眼睛看到的深度没有对齐，不移动头就没问题，一旦移动头部，就会轻微感觉左右眼不重叠并且出现拉丝。

**总之，该项目的无论单目还是双目深度，跟原本的VR180版本相比，是劣化，稍微移动头部就会感觉近处画面拉丝。**

When watching VR180 3D videos, I believe everyone has noticed that if the head rotates in the roll direction (i.e., tilting the head), the nearby images will overlap, resulting in a diminished viewing experience. Moreover, VR180 does not allow for movement in the forward, backward, left, and right directions, meaning it only has 3 degrees of freedom (3dof) instead of 6 degrees of freedom (6dof). Therefore, to further enhance the immersion experience, a type of 4DGS video has recently become popular, allowing for free movement in space and the ability to tilt the head as desired.

However, such videos require a large number of cameras to shoot, which is simply beyond the reach of ordinary people. It would be great if VR180 3D videos could be converted into 4DGS. Research on obtaining depth from binocular images has already matured (even depth obtained from 2D images has been achieved), and there are also methods for converting monocular images into 4DGS, but there is indeed no method for converting binocular images into 4DGS.

So I settled for the next best thing and used a classic method of combining a spherical shell with bump mapping to play back RGB+depth images. The binocular depth acquisition model I chose is Stereo Any Video from 2025, which has good temporal stability. The c-fast-foundationstereo from 2026 is quite fast, but it lacks temporal stability. Stereo Any Video requires inputting many frames for sliding window purposes, which can lead to excessive memory usage. Currently, I am using a resolution of 640x640, with a sliding window of 20 frames, and 12GB of VRAM is sufficient.

How about the playback effect? I can only say it's somewhat useful. The head can rotate significantly and move slightly, maintaining a three-dimensional sensation throughout. However, if the head moves too far, the "cliff" between depths will become severely distorted, affecting the viewing experience.

For the storage of the depth map, I used an upper and lower limit, logarithmic scaling, and 10-bit HEVC video encoding to ensure accuracy down to half a centimeter at close range. However, it seems to me that the model itself may not have such high accuracy

Moreover, this method converts the binocular video into left monocular + depth, doesn't that mean the RGB information on the right side will be lost? If the left eye aligns with the original left eye view, the image viewed by the right eye will inevitably be shifted, resulting in streaking.

I tried flipping the left and right images, then obtaining the depth of the right eye, and merging them into one video. The issue of blurring in the right eye was resolved, but the depths seen by the two eyes were not aligned. If I didn't move my head, everything was fine, but once I moved it, I could slightly feel that the left and right eyes did not overlap and there was a pulling sensation.

**In summary, whether it's monocular or binocular depth, compared to the original VR180 version, it's a downgrade. Even a slight movement of the head can cause the near image to appear blurry.**

## 项目结构 Layout
- `sav_runner.py`：模型加载、滑窗推理、球面视差→距离换算

  `sav_runner.py`: model loading, sliding-window inference and spherical disparity-to-range conversion

- `infer_sav.py`：单帧推理 → 深度图 / 预览图 / PLY 点云

  `infer_sav.py`: single-frame inference to depth map, preview and PLY point cloud

- `process_video.py`：视频或帧序列 → 左视图 + luma2 深度 SBS 的 H.265 视频（同时写 sidecar json）

  `process_video.py`: video or frame folder to an H.265 SBS clip with the left view plus luma2-packed depth (writes the sidecar json too)

- `gui_export.pyw`：图形界面，选源视频 + 时间段 → 依次导出单目彩色片段与深度视频

  `gui_export.pyw`: desktop GUI that picks a source video plus a time range and exports the mono colour clip and the depth video in sequence

- `mono8_drop.py`：SBS视频提取左半边。

  `mono8_drop.py`: Extract the left half of the SBS video.

- `view_depth_npy.pyw`：深度 npy 预览

  `view_depth_npy.pyw`: depth `.npy` previewer

- `third_party/stereoanyvideo/`：官方模型代码（Apache-2.0）与 VDA 权重

  `third_party/stereoanyvideo/`: official model code (Apache-2.0) and the VDA weights

## 环境要求 Requirements
- Windows 10/11，Python 3.10+

- NVIDIA 显卡；实测RTX 3060用640x640分辨率20步窗口占用显存约 9.0 GB，每帧3.6秒。

  NVIDIA graphics card; measured RTX 3060 uses approximately 9.0 GB of VRAM with a 640x640 resolution and 20-step window

- PyTorch 2.11.0+cu128，CUDA 12.x 

- ffmpeg / ffprobe with the libx265 encoder

## 安装 Installation
1. 下载该仓库，并执行`git submodule update`.

   Download the repository and run `git submodule update`.

2. 按照[stereoanyvideo](https://github.com/TomTomTommi/stereoanyvideo)要求安装。

   And follow [stereoanyvideo](https://github.com/TomTomTommi/stereoanyvideo)'s instructions to install.
   
   如果depth_anything权重下载不下来，可以用这个`https://hf-mirror.com/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth`，下载到`third_party\stereoanyvideo\models\Video-Depth-Anything\checkpoints\video_depth_anything_vits.pth`

4. 下载[主权重](https://drive.google.com/drive/folders/1c7L065dcBWhCYYjWYo2edGOG605PnpXv)到 `weights/StereoAnyVideo_MIX.pth`

   Download the [main checkpoint](https://drive.google.com/drive/folders/1c7L065dcBWhCYYjWYo2edGOG605PnpXv) to `weights/StereoAnyVideo_MIX.pth`
   
   国内可以用[百度云](https://pan.baidu.com/s/11Wm3jQNZmhRY_a0rG8ovbw?pwd=c2yd)。

5. 安装 ffmpeg（需含 H265解码器）

   Install ffmpeg (the build must have H265 decoder)
   
   Windows: `winget install Gyan.FFmpeg`
   
   检查 Check:  `ffmpeg -hide_banner -encoders | Select-String 265`



模型代码直接 import `core/stereoanyvideo.py`，不需要官方 README 里的 `pytorch3d`、`cub` 与 `pytorch_lightning`。
The model is imported straight from `core/stereoanyvideo.py`, so `pytorch3d`, `cub` and `pytorch_lightning` from the official README are not needed.

BF16 混合精度需要两处修补：`corr.py` 里 grid 与图像 dtype 对齐，`stereoanyvideo.py` 的凸上采样强制 FP32（unfoldNd 的分组 conv3d 在 BF16 下会溢出），仓库内已打好补丁。
BF16 mixed precision needs two patches: the grid is cast to the image dtype in `corr.py`, and the convex upsample in `stereoanyvideo.py` is forced to FP32 (unfoldNd's grouped conv3d overflows in BF16); both are already applied here.

6. 安装 xformer 和 triton，用于提速

   install xformer and triton to speed up
   
   Windows: `pip install xformer triton-windows`



## 导出 export

点击 gui_export.pyw，开启导出的图形界面。

Click on gui_export.pyw to open the graphical interface for export.

- 界面语言默认跟随系统语言，可用 `--lang zh` / `--lang en` 强制指定

  The UI language follows the system by default; force it with `--lang zh` or `--lang en`.

- 选择源视频（左右拼接的 VR180 3D SBS），填起始与结束时间，导出与源同分辨率的深度视频

  Pick the source VR180 3D SBS video, enter the start and end time, and export a depth video at the source resolution

- 勾选「导出单目 8bit 视频」时，先用 ffmpeg 按时间段裁出左半幅单目视频，深度推理跑同一时间段，导出会附带该片段

  With **'Export a left-eye mono'** ticked, ffmpeg first cuts a left-eye mono clip for the time range and depth inference runs on the same range; the clip ships with the export

- 导出完成后在同目录生成同名 `.json`，可直接拖进播放器

  The export leaves a sibling `.json` in the output folder, ready to open in the player

- 非图形界面导出可以用 process_video.py，例如`python process_video.py --input-video "a.mp4" --start 0 --end 5 --out a_depth.mp4`

  For non-graphical interface export, you can use process_video.py, for example, `python process_video.py --input-video "a.mp4" --start 0 --end 5 --out a_depth.mp4`



## 许可 License

MIT License。第三方模型 Stereo Any Video 为 Apache-2.0，Video Depth Anything 请遵循其原始许可。
MIT License. The third-party Stereo Any Video model is Apache-2.0; follow the original license of Video Depth Anything.
