"""GFPGAN / CodeFormer 人脸修复后处理。"""
import os
import torch
import numpy as np
from torchvision.transforms.functional import normalize

from setup_vendor import setup_gfpgan_paths, setup_codeformer_paths

GFPGAN_URL = (
    "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
)
CODEFORMER_URL = (
    "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"
)


def _device_from_str(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class GFPGANRestorer:
    """GFPGAN 人脸清晰化。"""

    def __init__(self, models_dir: str = "models", arch: str = "clean", device: str = "auto"):
        setup_gfpgan_paths()
        from gfpgan.utils import GFPGANer
        from basicsr.utils.download_util import load_file_from_url

        self.device = _device_from_str(device)
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
        if not os.path.isfile(model_path):
            model_path = load_file_from_url(
                url=GFPGAN_URL,
                model_dir=models_dir,
                progress=True,
                file_name="GFPGANv1.4.pth",
            )
        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=1,
            arch=arch,
            channel_multiplier=2,
            bg_upsampler=None,
            device=self.device,
        )

    def restore(self, image: np.ndarray) -> np.ndarray:
        _, _, output = self.restorer.enhance(
            image,
            has_aligned=False,
            only_center_face=True,
            paste_back=True,
        )
        return output


class CodeFormerRestorer:
    """CodeFormer 人脸清晰化。"""

    def __init__(
        self,
        models_dir: str = "models",
        fidelity: float = 0.5,
        device: str = "auto",
    ):
        setup_codeformer_paths()
        from basicsr.utils import img2tensor, tensor2img
        from basicsr.utils.download_util import load_file_from_url
        from basicsr.utils.registry import ARCH_REGISTRY
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self._img2tensor = img2tensor
        self._tensor2img = tensor2img
        self.fidelity = fidelity
        self.device = _device_from_str(device)
        os.makedirs(models_dir, exist_ok=True)

        ckpt = os.path.join(models_dir, "codeformer.pth")
        if not os.path.isfile(ckpt):
            ckpt = load_file_from_url(
                url=CODEFORMER_URL,
                model_dir=models_dir,
                progress=True,
                file_name="codeformer.pth",
            )

        self.net = (
            ARCH_REGISTRY.get("CodeFormer")(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
            .to(self.device)
        )
        checkpoint = torch.load(ckpt, map_location="cpu")["params_ema"]
        self.net.load_state_dict(checkpoint)
        self.net.eval()

        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
        )

    def restore(self, image: np.ndarray) -> np.ndarray:
        self.face_helper.clean_all()
        self.face_helper.read_image(image)
        if self.face_helper.get_face_landmarks_5(only_center_face=True, resize=640) == 0:
            return image
        self.face_helper.align_warp_face()

        for cropped_face in self.face_helper.cropped_faces:
            cropped_t = self._img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(cropped_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_t = cropped_t.unsqueeze(0).to(self.device)
            try:
                with torch.no_grad():
                    output = self.net(cropped_t, w=self.fidelity, adain=True)[0]
                restored = self._tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            except Exception:
                restored = self._tensor2img(
                    cropped_t, rgb2bgr=True, min_max=(-1, 1)
                )
            self.face_helper.add_restored_face(restored.astype("uint8"))

        self.face_helper.get_inverse_affine(None)
        return self.face_helper.paste_faces_to_input_image(upsample_img=image)


def create_restorer(method: str, **kwargs):
    method = (method or "none").lower()
    if method in ("none", "off", ""):
        return None
    if method == "gfpgan":
        return GFPGANRestorer(**kwargs)
    if method == "codeformer":
        return CodeFormerRestorer(**kwargs)
    raise ValueError(f"未知修复方法: {method}，可选 none / gfpgan / codeformer")
