"""InsightFace inswapper 深度学习换脸（roop / FaceFusion 同款核心模型）。"""
import os
import cv2
import insightface
from insightface.app import FaceAnalysis

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_device(device: str) -> tuple[bool, list[str]]:
    if device == "auto":
        try:
            import onnxruntime as ort
            use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            use_gpu = False
    else:
        use_gpu = device == "cuda"
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    return use_gpu, providers


class InsightFaceSwapper:
    """使用 InsightFace buffalo_l + inswapper_128 进行换脸。"""

    def __init__(
        self,
        models_dir: str = "models",
        face_analyser: str = "buffalo_l",
        swapper_name: str = "inswapper_128.onnx",
        det_size: tuple[int, int] = (640, 640),
        device: str = "auto",
    ):
        # InsightFace 约定: {root}/models/{name}/
        models_root = os.path.join(_PROJECT_ROOT, models_dir)
        os.makedirs(models_root, exist_ok=True)
        use_gpu, providers = _resolve_device(device)

        self.face_app = FaceAnalysis(
            name=face_analyser,
            root=_PROJECT_ROOT,
            providers=providers,
        )
        ctx_id = 0 if use_gpu else -1
        self.face_app.prepare(ctx_id=ctx_id, det_size=det_size)

        swapper_path = os.path.join(models_root, swapper_name)
        if not os.path.isfile(swapper_path):
            from download_models import download_inswapper
            download_inswapper(swapper_path)

        self.swapper = insightface.model_zoo.get_model(
            swapper_path,
            download=False,
            providers=providers,
        )
        self.source_face = None

    @staticmethod
    def _largest_face(faces):
        if not faces:
            return None
        return max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

    def set_source(self, image_path: str) -> None:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取源人脸图片: {image_path}")
        face = self._largest_face(self.face_app.get(img))
        if face is None:
            raise ValueError(f"源图片中未检测到人脸: {image_path}")
        self.source_face = face

    def swap_frame(self, frame):
        if self.source_face is None:
            raise RuntimeError("请先调用 set_source() 设置源人脸")
        target = self._largest_face(self.face_app.get(frame))
        if target is None:
            return frame
        return self.swapper.get(
            frame, target, self.source_face, paste_back=True
        )
