# change_face_in_video_RL

## 深度学习换脸（推荐）

使用 **InsightFace inswapper**（与 roop / FaceFusion 同类）+ **GFPGAN / CodeFormer** 人脸修复。

# AI 换脸项目说明

## 项目概述

本项目基于 InsightFace inswapper 模型实现视频人脸替换，支持 GFPGAN / CodeFormer 人脸修复。

## 核心文件

| 文件 | 作用 |
|------|------|
| `main.py` | 主程序入口 |
| `config.py` | 配置文件 |
| `PythonCv2.py` | 视频帧提取 |
| `newvideo.py` | 帧合并为视频 |
| `deletecache.py` | 缓存清理 |
| `dl_swap.py` | InsightFace 换脸核心 |
| `dl_restore.py` | 人脸修复（GFPGAN / CodeFormer） |
| `download_models.py` | 模型下载 |

## 工作流程

```
1. 配置 config.py
   - 设置源人脸图片（SOURCE_FACE）
   - 设置输入视频（VIDEO_INPUT）
   - 选择修复方法（RESTORE_METHOD: none/gfpgan/codeformer）

2. main.py 运行步骤：
   ┌─────────────────────────────────────────┐
   │ Step 1: 拆分视频帧                        │
   │   video2img() → 将视频拆分为图片帧        │
   │   输出到 origin/ 目录                     │
   ├─────────────────────────────────────────┤
   │ Step 2: 加载 InsightFace 模型             │
   │   加载 buffalo_l 人脸检测模型             │
   │   加载 inswapper_128 换脸模型             │
   ├─────────────────────────────────────────┤
   │ Step 3: 加载修复模型（可选）               │
   │   GFPGAN 或 CodeFormer                   │
   ├─────────────────────────────────────────┤
   │ Step 4: 逐帧处理                          │
   │   对每帧图像：                            │
   │     - 检测人脸位置                        │
   │     - 执行人脸替换                        │
   │     - 可选：人脸修复提升清晰度             │
   │   输出到 transfer/ 目录                   │
   ├─────────────────────────────────────────┤
   │ Step 5: 合并视频                          │
   │   mergevideo() → 将帧合成为视频           │
   │   输出 saveVideo.mp4                     │
   ├─────────────────────────────────────────┤
   │ Step 6: 清理缓存（可选）                   │
   │   删除 origin/ 和 transfer/ 目录          │
   └─────────────────────────────────────────┘

3. 运行命令
   python main.py
```

## 依赖说明

- **insightface**: 人脸检测与分析
- **onnxruntime**: ONNX 模型推理加速
- **opencv-python**: 视频/图像处理
- **GFPGAN / CodeFormer**: 人脸修复（可选）
- **basicsr**: 底层图像处理工具

## 硬件要求

- 有 NVIDIA 显卡时设置 `DEVICE = "cuda"` 可显著加速
- 无显卡时使用 CPU 模式运行

## 输出

- 换脸后视频：`saveVideo.mp4`（默认）
