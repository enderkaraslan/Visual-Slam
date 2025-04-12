import cv2
import numpy as np
import torch
import os
from depth_estimation.depth_anything_v2.dpt import DepthAnythingV2
from PIL import Image
import open3d as o3d

# Kamera parametreleri
FOCAL_LENGTH_X = 310.29173028#585.9846102
FOCAL_LENGTH_Y = 309.71260043#586.40459836
CX, CY = 156.66567722, 130.41321493#307.83648279, 257.87358272
MAX_DEPTH = 20
width, height = 320, 240 #640, 480

class DepthEstimator:
    def __init__(self, encoder):
        self.device = self._get_device()
        print(f"Kullanılan cihaz: {self.device}")
        self.model = self._load_model(encoder)

    def _get_device(self):
        """Mevcut en iyi cihazı döndür."""
        return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    def _load_model(self, encoder):
        """Derinlik modelini yükler."""
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        if encoder not in model_configs:
            raise ValueError(f"Geçersiz encoder tipi: {encoder}. Desteklenenler: {list(model_configs.keys())}")

        model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': MAX_DEPTH})
        model_path = f'/home/ender/Desktop/dev_ws/src/depth_estimation/depth_estimation/checkpoints/depth_anything_v2_metric_hypersim_{encoder}.pth'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model.to(self.device).eval()

    def process_image(self, raw_image):
        """Görüntüyü işleyerek derinlik haritası ve nokta bulutu oluşturur."""
        raw_image = cv2.resize(raw_image, (width, height))
        color_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        
        # Model ile derinlik tahmini yap
        depth_map = self.model.infer_image(raw_image, height)
        depth_map_resized = np.array(Image.fromarray(depth_map).resize((width, height), Image.NEAREST))
        
        # Nokta bulutu hesapla
        pcd = self._generate_point_cloud(depth_map_resized, color_image, width, height)

        return pcd

    def _generate_point_cloud(self, depth_map, color_image, width, height):
        """Derinlik haritasından nokta bulutu oluşturur."""
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x = (x - CX) / FOCAL_LENGTH_X
        y = (y - CY) / FOCAL_LENGTH_Y
        z = depth_map.astype(np.float32)
                
        points = np.stack((x * z, y * z, z), axis=-1).reshape(-1, 3)
        colors = (np.array(color_image).reshape(-1, 3) / 255.0).astype(np.float32)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    

