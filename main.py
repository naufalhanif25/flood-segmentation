import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

class TestModel:
    def __init__(self, image_size = 256):
        self.image_size = image_size
        self.infer_transform = A.Compose([
            A.Resize(self.image_size, self.image_size),
            A.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def _overlay_mask(self, image, mask, alpha = 0.5, color = (255, 0, 0)):
        overlay = image.copy()
        mask = mask.astype(bool)
        overlay[mask] = color

        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    def infer_flood_image(self, image_path, model, device, threshold = 0.5, show = True, save_dir = None):
        image_bgr = cv2.imread(image_path)

        if image_bgr is None:
            raise ValueError(f"Image not found: {image_path}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        transformed = self.infer_transform(image = image_rgb)
        input_tensor = transformed["image"].unsqueeze(0).to(device)

        model.eval()

        with torch.no_grad():
            preds = model(input_tensor)
            preds = torch.sigmoid(preds)
            mask = (preds > threshold).float()

        mask = mask.squeeze().cpu().numpy()
        mask = cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]))
        mask = (mask > 0.5).astype(np.uint8)
        overlay = self._overlay_mask(image_rgb, mask)

        if save_dir:
            os.makedirs(save_dir, exist_ok = True)
            
            cv2.imwrite(os.path.join(save_dir, "original.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(save_dir, "mask.png"), mask * 255)
            cv2.imwrite(os.path.join(save_dir, "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        if show:
            fig, axs = plt.subplots(1, 3, figsize = (18, 6))

            axs[0].imshow(image_rgb)
            axs[0].set_title("Original Image")
            axs[0].axis("off")

            axs[1].imshow(mask, cmap = "gray")
            axs[1].set_title("Predicted Flood Mask")
            axs[1].axis("off")

            axs[2].imshow(overlay)
            axs[2].set_title("Flood Segmentation Overlay")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()

        return overlay, mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Program untuk melakukan inferensi segmentasi banjir pada gambar. \nDapat memproses satu gambar dan menampilkan serta menyimpan hasil segmentasi.", 
        usage = "program [options...] [args...]", add_help = False
    )

    parser.add_argument("--image", type = str, default = "test/image.png", help = "Path gambar yang digunakan untuk testing")
    parser.add_argument("--outdir", type = str, default = None, required = False, help = "Path direktori output untuk menyimpan hasil testing")
    parser.add_argument("-h", "--help", help = "Tampilkan pesan bantuan", action = "help")
    parser.add_argument("-v", "--version", action = "version", version = "v1.0.0", help = "Tampilkan versi program")

    args = parser.parse_args()
    test_model = TestModel(256)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = smp.Unet(
        encoder_name = "resnet34",
        encoder_weights = "imagenet",
        in_channels = 3,
        classes = 1
    )

    model.load_state_dict(torch.load("model/best_flood_segmentation.pth", map_location = device))
    model.to(device)

    test_model.infer_flood_image(
        args.image,
        model,
        device,
        save_dir = args.outdir
    )

    