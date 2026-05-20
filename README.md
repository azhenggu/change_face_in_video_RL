# change_face_in_video_RL
change face in video with InsightFace inswapper and GFPGAN / CodeFormer
# AI-Change-face-in-the-video

## 深度学习换脸（推荐）

使用 **InsightFace inswapper**（与 roop / FaceFusion 同类）+ **GFPGAN / CodeFormer** 人脸修复，效果远好于下方 dlib 68 点方案。

### 快速开始

```powershell
# 1. 安装依赖（Windows Python 3.13）
.\install_deps.ps1

# 2. 下载 inswapper 模型（若 main 报错可手动执行）
python download_models.py

# 3. 修改 config.py：SOURCE_FACE、VIDEO_INPUT、RESTORE_METHOD 等

# 4. 运行
python main.py
```

| 配置项 | 说明 |
|--------|------|
| `SOURCE_FACE` | 源人脸图片，如 `1.png` |需要放入文件夹
| `VIDEO_INPUT` | 输入视频 |
| `RESTORE_METHOD` | `none` / `gfpgan` / `codeformer` |
| `CODEFORMER_FIDELITY` | 0~1，越大越保真、越小越清晰 |

- 输出：`saveVideo.mp4`
- 有 NVIDIA 显卡时可在 `config.py` 设 `DEVICE = "cuda"` 显著加速

---



