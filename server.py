"""
NeuroScan v2 — Brain MRI Classification Backend
Run:  python server.py
Open: http://127.0.0.1:5000
"""

import io, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Config 
MODEL_PATH  = "brain_mri_checkpoint.pth"   # ← change to your .pth filename
IMG_SIZE    = 224
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model 
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if pool: layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class BrainMRINet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32,  True),  ConvBlock(32,  64,  True),
            ConvBlock(64,  128, True),  ConvBlock(128, 256, False),
            ConvBlock(256, 256, True),  ConvBlock(256, 512, False),
            ConvBlock(512, 512, True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(512, 256),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(256, num_classes),
        )
    def forward(self, x):
        x = self.features(x); x = self.gap(x)
        return self.classifier(x.view(x.size(0), -1))

# ── Load 
def load_model():
    model = BrainMRINet(NUM_CLASSES)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"\n '{MODEL_PATH}' not found in: {os.getcwd()}\n"
            f"   Files here: {os.listdir('.')}"
        )
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f" Checkpoint loaded (val_acc={ckpt.get('best_val_acc','N/A')})")
    else:
        model.load_state_dict(ckpt)
        print(" State dict loaded")
    model.to(DEVICE).eval()
    return model

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# ── Flask 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app      = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)
model    = load_model()

@app.route("/")
def index(): return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:f>")
def static_files(f): return send_from_directory(BASE_DIR, f)

@app.route("/health")
def health(): return jsonify({"status": "ok", "device": str(DEVICE)}), 200

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    ext  = os.path.splitext(file.filename)[1].lower()
    if ext not in {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}:
        return jsonify({"error": f"Unsupported type: {ext}"}), 400
    try:
        img    = Image.open(io.BytesIO(file.read())).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)
    except Exception as e:
        return jsonify({"error": str(e)}), 422

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).squeeze()
        pred  = int(probs.argmax())
        conf  = float(probs[pred])

    return jsonify({
        "predicted_class": CLASS_NAMES[pred],
        "confidence"     : round(conf, 4),
        "probabilities"  : {c: round(float(probs[i]),4) for i,c in enumerate(CLASS_NAMES)},
    }), 200

if __name__ == "__main__":
    print(f"\n{'='*48}\n   NeuroScan v2 — Brain MRI Classifier")
    print(f"   Open: http://127.0.0.1:5000\n  Device: {DEVICE}\n{'='*48}\n")
    app.run(host="127.0.0.1", port=5000, debug=False)