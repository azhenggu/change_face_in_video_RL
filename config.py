"""深度学习换脸流水线配置。"""
import os

# 路径（相对项目根目录）
SOURCE_FACE = "1.png"           # 源人脸图片
VIDEO_INPUT = "video.mp4"       # 输入视频
OUTPUT_VIDEO = "saveVideo.mp4" # 输出视频

ORIGIN_DIR = "origin/"
TRANSFER_DIR = "transfer/"
MODELS_DIR = "models/"

# InsightFace
FACE_ANALYSER = "buffalo_l"     # 人脸检测/特征模型
SWAPPER_MODEL = "inswapper_128.onnx"
DET_SIZE = (640, 640)

# 人脸修复: "none" | "gfpgan" | "codeformer"
RESTORE_METHOD = "codeformer"
# CodeFormer 保真度 0~1，越大越接近原脸、越小越清晰
CODEFORMER_FIDELITY = 0.5
# GFPGAN 架构: clean | original
GFPGAN_ARCH = "clean"

# 设备: "auto" | "cuda" | "cpu"
DEVICE = "auto"

# 视频
OUTPUT_FPS = 0                  # 0 = 使用源视频帧率
FRAME_EXT = ".jpg"              # 中间帧格式，.jpg 或 .png
JPEG_QUALITY = 95

# 是否在结束后询问删除缓存
ASK_DELETE_CACHE = True
