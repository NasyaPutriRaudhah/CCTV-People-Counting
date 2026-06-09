from ultralytics import YOLO
from config import MODEL_PATH, DEVICE

def load_models():
    print(f"Loading models from {MODEL_PATH}...")
    
    if MODEL_PATH.endswith('.engine'):
        # TensorRT - tidak perlu .to(device)
        model1 = YOLO(MODEL_PATH)
        model2 = YOLO(MODEL_PATH)
    else:
        # PyTorch biasa
        model1 = YOLO(MODEL_PATH).to(DEVICE)
        model2 = YOLO(MODEL_PATH).to(DEVICE)
    
    print("✓ Models loaded successfully")
    return model1, model2

