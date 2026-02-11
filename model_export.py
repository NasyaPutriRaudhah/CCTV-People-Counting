from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model
model.export(format="openvino")  # creates 'yolo26n_openvino_model/'
