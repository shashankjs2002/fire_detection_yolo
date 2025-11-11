# app.py - Reverted to Frame-by-Frame Base64 Streaming for Reliability
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import threading
import time
from collections import defaultdict, deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime
import os
from typing import Dict, List, Tuple, Optional

# Configure logging
# logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Suppress SocketIO specific loggers
# logging.getLogger('socketio').setLevel(logging.WARNING)
# logging.getLogger('engineio').setLevel(logging.WARNING)
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-prod')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=False)

# Load custom YOLOv8n model (num_classes=1). Ensure 'best.pt' is in the project root or update path.
MODEL_PATH = 'best.pt'
try:
    model = YOLO(MODEL_PATH)
    logger.info(f"Loaded YOLO model from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Configuration
ADMIN_EMAIL = os.environ.get('ADMIN_EMAIL', 'admin@example.com')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', 'your-email@gmail.com')
SMTP_PASS = os.environ.get('SMTP_PASS', 'your-app-password')

# State management
clients: Dict[str, Dict] = {}  # {client_id: {'frame': str, 'detections': List, 'consecutive_detections': int, 'last_detection_time': float}}
detection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=5))  # {client_id: deque of bools (detection in frame)}
alert_cooldown: Dict[str, float] = {}  # {client_id: last_alert_time} to prevent spam

CONFIDENCE_THRESHOLD = 0.1
ALERT_CONSECUTIVE_FRAMES = 5
ALERT_COOLDOWN_SEC = 60  # Seconds between alerts per client
FRAME_RATE = 1  # Frames per second for streaming (to reduce load, ~100ms interval)

def send_email_alert(client_id: str, consecutive: int):
    """Send email alert asynchronously."""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USER
        msg['To'] = ADMIN_EMAIL
        msg['Subject'] = f"Object Detection Alert: Client {client_id} - {consecutive} Consecutive Frames"
        body = f"""
        Alert Timestamp: {datetime.now().isoformat()}
        Client ID: {client_id}
        Detection: Object (class 0) detected in {consecutive} consecutive frames.
        Please check the admin panel for details.
        """
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_USER, ADMIN_EMAIL, msg.as_string())
        server.quit()
        logger.info(f"Email alert sent for client {client_id}")
    except Exception as e:
        logger.error(f"Failed to send email for client {client_id}: {e}")

def process_frame(client_id: str, data_uri: str) -> Optional[str]:
    """Process a single frame: decode, detect, annotate, and return base64 encoded annotated frame."""
    if model is None:
        logger.error("Model not loaded")
        return None

    try:
        # Decode base64
        if data_uri.startswith('data:image'):
            header, encoded = data_uri.split(',', 1)
        else:
            encoded = data_uri
        frame_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None or frame.size == 0:
            logger.warning(f"Invalid frame received from {client_id}")
            return None

        # Run inference
        print("frame-->", frame)
        results = model(frame, verbose=False, conf=CONFIDENCE_THRESHOLD, save=True)
        # print(f"results --> {results}")
        # Extract detections (single class assumed: 0)
        detections = []
        detection_detected = False
        for r in results:
            if r.boxes is not None:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].cpu().numpy().item()
                    cls = int(box.cls[0].cpu().numpy().item())
                    if conf >= CONFIDENCE_THRESHOLD:
                        detections.append((x1, y1, x2, y2, conf, cls))
                        detection_detected = True

        # Update history
        detection_history[client_id].append(detection_detected)
        consecutive = sum(detection_history[client_id])
        
        # Check for alert
        current_time = time.time()
        last_alert = alert_cooldown.get(client_id, 0)
        if consecutive >= ALERT_CONSECUTIVE_FRAMES and (current_time - last_alert) > ALERT_COOLDOWN_SEC:
            alert_cooldown[client_id] = current_time
            threading.Thread(target=send_email_alert, args=(client_id, consecutive), daemon=True).start()
            # Emit alert to admin for UI notification
            emit('alert', {
                'client_id': client_id,
                'consecutive': consecutive
            }, room='admin')

        # Update client state
        clients[client_id].update({
            'detections': detections,
            'consecutive_detections': consecutive,
            'last_detection_time': current_time
        })

        # Annotate frame
        annotated_frame = results[0].plot() if len(results) > 0 else frame

        # Encode to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')
        processed_uri = f"data:image/jpeg;base64,{jpg_as_text}"
        clients[client_id]['frame'] = processed_uri

        logger.debug(f"Processed frame for {client_id}: {len(detections)} detections, {consecutive} consecutive")
        print(f"Processed frame for {client_id}: {len(detections)} detections, {consecutive} consecutive")
        return processed_uri

    except Exception as e:
        logger.error(f"Error processing frame for {client_id}: {e}")
        return None

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    clients[client_id] = {
        'frame': None,
        'detections': [],
        'consecutive_detections': 0,
        'last_detection_time': 0
    }
    detection_history[client_id] = deque(maxlen=ALERT_CONSECUTIVE_FRAMES)
    logger.info(f"Client {client_id} connected")
    emit('status', {'msg': 'Connected successfully', 'client_id': client_id})

@socketio.on('frame')
def handle_frame(data):
    client_id = request.sid
    if client_id not in clients:
        return

    frame_data = data.get('frame')
    if not frame_data:
        logger.warning(f"No frame data from {client_id}")
        return

    # Optional: Frame rate limiting (skip if too recent)
    last_process = clients[client_id].get('last_process', 0)
    if time.time() - last_process < 1.0 / FRAME_RATE:
        return
    clients[client_id]['last_process'] = time.time()

    processed = process_frame(client_id, frame_data)
    if processed:
        emit('processed_frame', {
            'frame': processed,
            'client_id': client_id,
            'detections': clients[client_id]['detections'],
            'consecutive_detections': clients[client_id]['consecutive_detections']
        }, room='admin')

@socketio.on('join_admin')
def join_admin():
    join_room('admin')
    logger.info(f"Admin joined: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    if client_id in clients:
        del clients[client_id]
    if client_id in detection_history:
        del detection_history[client_id]
    if client_id in alert_cooldown:
        del alert_cooldown[client_id]
    logger.info(f"Client {client_id} disconnected")

# Optional: Health check endpoint
@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None, 'clients': len(clients)})

# Routes
@app.route('/')
def client_view():
    return render_template('index.html')

@app.route('/admin')
def admin_view():
    return render_template('admin.html')

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))