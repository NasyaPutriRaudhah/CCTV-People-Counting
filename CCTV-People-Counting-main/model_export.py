from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo11n.pt")

# Export the model
model.export(format="engine", half=True)  # creates 'yolo26n_openvino_model/'
