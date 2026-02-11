import cv2
import numpy as np
import socket
import pickle
import struct
import threading
import queue
import time
from ultralytics import YOLO
import torch

# =========================
# SERVER CONFIG
# =========================
SERVER_HOST = '0.0.0.0'  # Listen on all interfaces
SERVER_PORT = 9999
MODEL_PATH = 'yolo26n.pt'
CONF_THRESHOLD = 0.2
IOU_THRESHOLD = 0.65

# GPU Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\n{'='*70}")
print(f"GPU INFERENCE SERVER")
print(f"{'='*70}")
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
print(f"{'='*70}\n")

# =========================
# LOAD MODELS ON GPU
# =========================
print("Loading YOLO models on GPU...")
model_v1 = YOLO(MODEL_PATH)
model_v1.to(DEVICE)
print(f"✓ Model 1 loaded on {DEVICE}")

model_v2 = YOLO(MODEL_PATH)
model_v2.to(DEVICE)
print(f"✓ Model 2 loaded on {DEVICE}")

# =========================
# INFERENCE FUNCTIONS
# =========================
def run_inference_v1(frame):
    """Run inference for Video 1 (ROI detection with tracking)"""
    start_time = time.time()
    
    results = model_v1.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
        device=DEVICE
    )
    
    inference_time = (time.time() - start_time) * 1000
    
    # Extract results
    detections = {
        'ids': None,
        'boxes': None,
        'confs': None,
        'inference_time': inference_time
    }
    
    if results[0].boxes.id is not None:
        detections['ids'] = results[0].boxes.id.cpu().numpy().astype(int)
        detections['boxes'] = results[0].boxes.xyxy.cpu().numpy().astype(int)
        detections['confs'] = results[0].boxes.conf.cpu().numpy()
    
    return detections

def run_inference_v2(frame):
    """Run inference for Video 2 (line crossing)"""
    start_time = time.time()
    
    results = model_v2(
        frame,
        classes=[0],
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
        device=DEVICE
    )
    
    inference_time = (time.time() - start_time) * 1000
    
    # Extract results
    detections = {
        'boxes': None,
        'confs': None,
        'inference_time': inference_time
    }
    
    if len(results[0].boxes) > 0:
        detections['boxes'] = results[0].boxes.xyxy.cpu().numpy().astype(int)
        detections['confs'] = results[0].boxes.conf.cpu().numpy()
    
    return detections

# =========================
# CLIENT HANDLER
# =========================
class ClientHandler(threading.Thread):
    """Handle a single client connection"""
    def __init__(self, conn, addr, client_id):
        threading.Thread.__init__(self)
        self.conn = conn
        self.addr = addr
        self.client_id = client_id
        self.daemon = True
        self.stopped = False
        
        # Statistics
        self.frames_processed = 0
        self.total_inference_time = 0
        self.start_time = time.time()
        
    def recv_frame(self):
        """Receive frame from client"""
        try:
            # Receive data length
            data_size = struct.unpack(">L", self.conn.recv(4))[0]
            
            # Receive frame data
            frame_data = b""
            while len(frame_data) < data_size:
                packet = self.conn.recv(min(data_size - len(frame_data), 4096))
                if not packet:
                    return None
                frame_data += packet
            
            # Deserialize
            data = pickle.loads(frame_data)
            return data
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error receiving frame: {e}")
            return None
    
    def send_results(self, results):
        """Send inference results back to client"""
        try:
            # Serialize results
            data = pickle.dumps(results)
            
            # Send data length
            self.conn.sendall(struct.pack(">L", len(data)))
            
            # Send data
            self.conn.sendall(data)
            return True
            
        except Exception as e:
            print(f"[Client {self.client_id}] Error sending results: {e}")
            return False
    
    def run(self):
        print(f"[Client {self.client_id}] Connected from {self.addr}")
        
        try:
            while not self.stopped:
                # Receive frame request
                data = self.recv_frame()
                
                if data is None:
                    break
                
                video_id = data['video_id']
                frame = data['frame']
                
                # Run inference based on video ID
                if video_id == 1:
                    results = run_inference_v1(frame)
                elif video_id == 2:
                    results = run_inference_v2(frame)
                else:
                    results = {'error': 'Invalid video_id'}
                
                # Send results back
                if not self.send_results(results):
                    break
                
                # Update statistics
                self.frames_processed += 1
                if 'inference_time' in results:
                    self.total_inference_time += results['inference_time']
                
                # Print statistics every 100 frames
                if self.frames_processed % 100 == 0:
                    avg_inference = self.total_inference_time / self.frames_processed
                    elapsed = time.time() - self.start_time
                    fps = self.frames_processed / elapsed
                    print(f"[Client {self.client_id}] Processed: {self.frames_processed} | "
                          f"Avg inference: {avg_inference:.2f}ms | FPS: {fps:.2f}")
        
        except Exception as e:
            print(f"[Client {self.client_id}] Error: {e}")
        
        finally:
            self.conn.close()
            elapsed = time.time() - self.start_time
            avg_fps = self.frames_processed / elapsed if elapsed > 0 else 0
            print(f"[Client {self.client_id}] Disconnected | "
                  f"Total frames: {self.frames_processed} | "
                  f"Avg FPS: {avg_fps:.2f}")
    
    def stop(self):
        self.stopped = True

# =========================
# SERVER
# =========================
class InferenceServer:
    """Main server to accept client connections"""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = None
        self.clients = []
        self.client_id_counter = 0
        self.stopped = False
        
    def start(self):
        """Start the server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        print(f"\n✓ Server started on {self.host}:{self.port}")
        print(f"Waiting for connections...\n")
        
        try:
            while not self.stopped:
                try:
                    self.server_socket.settimeout(1.0)
                    conn, addr = self.server_socket.accept()
                    
                    # Create client handler
                    client_handler = ClientHandler(conn, addr, self.client_id_counter)
                    client_handler.start()
                    
                    self.clients.append(client_handler)
                    self.client_id_counter += 1
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.stopped:
                        print(f"Error accepting connection: {e}")
        
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server"""
        self.stopped = True
        
        # Stop all client handlers
        for client in self.clients:
            client.stop()
        
        # Close server socket
        if self.server_socket:
            self.server_socket.close()
        
        print("Server stopped")

# =========================
# MAIN
# =========================
def main():
    server = InferenceServer(SERVER_HOST, SERVER_PORT)
    server.start()

if __name__ == "__main__":
    main()