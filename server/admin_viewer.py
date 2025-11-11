# admin_viewer.py
"""
Python-based Admin Panel Viewer using OpenCV
Run this on the server to view all client streams in a desktop window
"""

import cv2
import numpy as np
import socketio
import base64
from threading import Thread, Lock
import time
from collections import OrderedDict

class AdminPanelViewer:
    def __init__(self, server_url='http://localhost:5000'):
        self.server_url = server_url
        self.sio = socketio.Client()
        self.frames = OrderedDict()
        self.lock = Lock()
        self.running = True
        self.window_name = 'Admin Panel - Object Detection Monitor'
        
        # Setup socket events
        self.setup_socket_events()
        
    def setup_socket_events(self):
        @self.sio.on('connect')
        def on_connect():
            print(f'Connected to server at {self.server_url}')
            self.request_frames()
        
        @self.sio.on('disconnect')
        def on_disconnect():
            print('Disconnected from server')
        
        @self.sio.on('admin_frames')
        def on_admin_frames(data):
            frames_data = data.get('frames', {})
            with self.lock:
                self.frames = OrderedDict()
                for client_id, frame_base64 in frames_data.items():
                    try:
                        # Decode base64 image
                        img_data = base64.b64decode(frame_base64.split(',')[1])
                        nparr = np.frombuffer(img_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            self.frames[client_id] = frame
                    except Exception as e:
                        print(f"Error decoding frame for {client_id}: {e}")
    
    def request_frames(self):
        """Continuously request frames from server"""
        while self.running:
            try:
                self.sio.emit('request_admin_frames')
                time.sleep(0.2)  # Request every 200ms
            except Exception as e:
                print(f"Error requesting frames: {e}")
                break
    
    def create_grid_display(self, frames_dict, grid_width=1920, grid_height=1080):
        """Create a grid display of all client frames"""
        if not frames_dict:
            # Create empty frame with message
            empty = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            text = "No active clients"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 2, 3)[0]
            text_x = (grid_width - text_size[0]) // 2
            text_y = (grid_height + text_size[1]) // 2
            cv2.putText(empty, text, (text_x, text_y), font, 2, (100, 100, 100), 3)
            return empty
        
        # Calculate grid dimensions
        num_frames = len(frames_dict)
        cols = int(np.ceil(np.sqrt(num_frames)))
        rows = int(np.ceil(num_frames / cols))
        
        # Calculate cell dimensions
        cell_width = grid_width // cols
        cell_height = grid_height // rows
        
        # Create grid canvas
        grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
        
        # Place frames in grid
        for idx, (client_id, frame) in enumerate(frames_dict.items()):
            row = idx // cols
            col = idx % cols
            
            # Resize frame to fit cell
            resized = cv2.resize(frame, (cell_width - 10, cell_height - 10))
            
            # Calculate position
            y_start = row * cell_height + 5
            x_start = col * cell_width + 5
            y_end = y_start + resized.shape[0]
            x_end = x_start + resized.shape[1]
            
            # Place frame in grid
            try:
                grid[y_start:y_end, x_start:x_end] = resized
                
                # Add border
                cv2.rectangle(grid, 
                            (x_start - 2, y_start - 2), 
                            (x_end + 2, y_end + 2), 
                            (0, 255, 0), 2)
            except Exception as e:
                print(f"Error placing frame in grid: {e}")
        
        # Add info bar at top
        info_bar_height = 60
        info_bar = np.zeros((info_bar_height, grid_width, 3), dtype=np.uint8)
        info_bar[:] = (30, 30, 30)
        
        # Add text
        cv2.putText(info_bar, f"Active Clients: {num_frames}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(info_bar, timestamp, (grid_width - 400, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        
        # Live indicator
        cv2.circle(info_bar, (grid_width - 450, 30), 10, (0, 255, 0), -1)
        cv2.putText(info_bar, "LIVE", (grid_width - 500, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Combine info bar and grid
        final_display = np.vstack([info_bar, grid])
        
        return final_display
    
    def display_loop(self):
        """Main display loop"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1920, 1080)
        
        print("\n" + "="*60)
        print("Admin Panel Viewer Started")
        print("="*60)
        print("Controls:")
        print("  - Press 'q' or ESC to quit")
        print("  - Press 'f' to toggle fullscreen")
        print("="*60 + "\n")
        
        fullscreen = False
        
        while self.running:
            with self.lock:
                frames_copy = self.frames.copy()
            
            # Create grid display
            display = self.create_grid_display(frames_copy)
            
            # Show display
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # q or ESC
                self.running = False
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 
                                        cv2.WINDOW_NORMAL)
        
        cv2.destroyAllWindows()
    
    def start(self):
        """Start the admin panel viewer"""
        try:
            # Connect to server
            print(f"Connecting to server at {self.server_url}...")
            self.sio.connect(self.server_url)
            
            # Start frame request thread
            request_thread = Thread(target=self.request_frames, daemon=True)
            request_thread.start()
            
            # Start display loop in main thread
            self.display_loop()
            
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.running = False
            if self.sio.connected:
                self.sio.disconnect()
            print("\nAdmin Panel Viewer stopped")


def main():
    # Configuration
    SERVER_URL = 'http://localhost:5000'  # Change to your server address
    
    # Create and start viewer
    viewer = AdminPanelViewer(server_url=SERVER_URL)
    viewer.start()


if __name__ == '__main__':
    main()