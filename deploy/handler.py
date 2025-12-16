import torch
import torchvision.transforms as T
from PIL import Image
import io
import os
import sys

# Add project root to path so we can import models/utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import StrokeClassifier
from utils.safety import check_saturation, mc_dropout_predict

class StrokeHandler:
    def __init__(self, model_path='best_model.pth', device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Handler initializing on device: {self.device}")
        
        # Load Model
        self.model = StrokeClassifier().to(self.device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"[INFO] Model loaded from {model_path}")
        else:
            print(f"[WARNING] Model path {model_path} not found.")
            
        self.model.eval()
        
        # Transforms (Must match Training!)
        # - Resize 512x512
        # - ToTensor
        # - Normalize ImageNet
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, image_bytes):
        """Converts bytes to Tensor"""
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Safety Check 1: Saturation (OOD)
        # We need the tensor *before* normalization for easy stats, or handle it carefully.
        # Let's use the utility which expects normalized tensor, roughly.
        tensor = self.transform(image)
        return tensor.unsqueeze(0).to(self.device) # Add Batch Dim

    def inference(self, tensor, threshold=0.5):
        """Runs prediction with Safety wrapper"""
        
        # Safety Check: Saturation
        # Note: tensor is [1, 3, 512, 512]
        is_safe = check_saturation(tensor.squeeze(0))
        
        if not is_safe:
            return {"error": "OOD Detected: Image appears too saturated for a CT scan."}

        # Clinical Uncertainty (MC Dropout)
        # standard_logits = self.model(tensor) # fast path
        
        # MC Dropout Path (Slower but gives uncertainty)
        # mean_prob is [1, 2] (Batch, Classes)
        mean_prob, uncertainty = mc_dropout_predict(self.model, tensor, n_samples=10)
        
        # Taking class 1 (Stroke) probability from the first (and only) item in batch
        stroke_prob = mean_prob[0, 1] 
        uncertainty_score = uncertainty[0, 1] 
            
        prediction = "Stroke" if stroke_prob > threshold else "Normal"
        
        return {
            "prediction": prediction,
            "confidence": round(float(stroke_prob), 4),
            "uncertainty": round(float(uncertainty_score), 4),
            "is_safe": True
        }

    def handle(self, data, threshold=0.5):
        """Main entry point for serving"""
        # data expected to be bytes
        try:
            tensor = self.preprocess(data)
            result = self.inference(tensor, threshold=threshold)
            return result
        except Exception as e:
            return {"error": str(e)}
