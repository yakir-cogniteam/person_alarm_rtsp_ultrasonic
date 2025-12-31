#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import paho.mqtt.client as mqtt
import threading
import time
import json
import numpy as np
import math


class Operator:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883, 
                 mqtt_topic="camera/control", mqtt_status_topic="camera/status",
                 mqtt_lidar_topic="lidar/scan",  # NEW: LiDAR scan topic
                 rtsp_url=None):
        """
        Initialize the Operator interface
        
        Args:
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            mqtt_topic: Topic to publish control commands
            mqtt_status_topic: Topic to subscribe for status updates
            mqtt_lidar_topic: Topic to subscribe for LiDAR scan data
            rtsp_url: RTSP stream URL from the camera
        """
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.mqtt_status_topic = mqtt_status_topic
        self.mqtt_lidar_topic = mqtt_lidar_topic  # NEW
        self.rtsp_url = rtsp_url
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Video capture
        self.video_capture = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        
        # Status dictionary from camera
        self.status_dict = {}
        self.status_lock = threading.Lock()
        
        # NEW: LiDAR scan data
        self.latest_lidar_scan = None
        self.lidar_scan_lock = threading.Lock()
        
        # View mode toggle
        self.view_mode = "rtsp"  # "rtsp" or "map"
        
        # Map visualization parameters
        self.map_image_size = 800
        self.map_center = self.map_image_size // 2
        self.map_scale = 100  # pixels per meter
        self.max_distance = 4.0  # meters
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Camera Operator Interface")
        self.root.geometry("1200x600")
        
        # Configure grid weights for responsive layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        
        self._setup_ui()
        self._setup_mqtt()
        
        # Bind keyboard events
        self.root.bind('<KeyPress>', self._on_key_press)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_ui(self):
        """Setup the Tkinter user interface"""
        
        # Left panel for buttons
        left_panel = tk.Frame(self.root, bg='#2c3e50', padx=10, pady=10)
        left_panel.grid(row=0, column=0, sticky='nsew')
        
        # Title for left panel
        title_label = tk.Label(
            left_panel, 
            text="Camera Controls", 
            font=('Arial', 14, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=(0, 20))
        
        # Button styling
        button_style = {
            'font': ('Arial', 12),
            'width': 15,
            'height': 2,
            'bg': '#3498db',
            'fg': 'white',
            'activebackground': '#2980b9',
            'activeforeground': 'white',
            'relief': 'raised',
            'bd': 3
        }
        
        # View Mode Toggle button
        self.view_mode_btn = tk.Button(
            left_panel,
            text="View Mode: RTSP",
            command=self._toggle_view_mode,
            font=('Arial', 12, 'bold'),
            width=15,
            height=2,
            bg='#9b59b6',
            fg='white',
            activebackground='#8e44ad',
            activeforeground='white',
            relief='raised',
            bd=3
        )
        self.view_mode_btn.pack(pady=10)
        
        # Separator
        separator1 = ttk.Separator(left_panel, orient='horizontal')
        separator1.pack(fill='x', pady=10)
        
        # Calibrate button
        self.calibrate_btn = tk.Button(
            left_panel,
            text="Calibrate",
            command=lambda: self._send_command("calibrate"),
            **button_style
        )
        self.calibrate_btn.pack(pady=10)
        
        # Go Home button
        self.home_btn = tk.Button(
            left_panel,
            text="Go Home",
            command=lambda: self._send_command("go_home"),
            **button_style
        )
        self.home_btn.pack(pady=10)
        
        # Switch State button
        self.switch_state_btn = tk.Button(
            left_panel,
            text="Switch State",
            command=lambda: self._send_command("switch_state"),
            **button_style
        )
        self.switch_state_btn.pack(pady=10)
        
        # Separator
        separator = ttk.Separator(left_panel, orient='horizontal')
        separator.pack(fill='x', pady=20)
        
        # Keyboard controls info
        info_label = tk.Label(
            left_panel,
            text="Keyboard Controls:",
            font=('Arial', 11, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        info_label.pack(pady=(0, 10))
        
        controls_text = """
        ↑ : Up
        ↓ : Down
        ← : Left
        → : Right
        """
        
        controls_label = tk.Label(
            left_panel,
            text=controls_text,
            font=('Arial', 10),
            bg='#2c3e50',
            fg='#ecf0f1',
            justify='left'
        )
        controls_label.pack()
        
        # MQTT status
        self.status_label = tk.Label(
            left_panel,
            text="MQTT: Disconnected",
            font=('Arial', 9),
            bg='#2c3e50',
            fg='#e74c3c'
        )
        self.status_label.pack(side='bottom', pady=10)
        
        # Center panel for video/map
        center_panel = tk.Frame(self.root, bg='black')
        center_panel.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        
        # Video/Map label
        self.video_label = tk.Label(center_panel, bg='black')
        self.video_label.pack(expand=True, fill='both')
        
        # Video info label
        self.video_info_label = tk.Label(
            center_panel,
            text="Waiting for video stream...",
            font=('Arial', 12),
            bg='black',
            fg='white'
        )
        self.video_info_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Right panel for status indicators
        right_panel = tk.Frame(self.root, bg='#34495e', padx=10, pady=10, width=200)
        right_panel.grid(row=0, column=2, sticky='nsew')
        right_panel.grid_propagate(False)  # Prevent frame from shrinking
        
        # Title for right panel
        status_title = tk.Label(
            right_panel,
            text="System Status",
            font=('Arial', 14, 'bold'),
            bg='#34495e',
            fg='white'
        )
        status_title.pack(pady=(0, 20))
        
        # Status LabelFrames (4 total)
        self.status_frames = {}
        
        # 1. Map Calibrated Status
        self.status_frames['is_map_calibrated'] = self._create_status_frame(
            right_panel,
            "Map Calibrated",
            "is_map_calibrated"
        )
        
        # 2. System State Status
        self.status_frames['system_state'] = self._create_status_frame(
            right_panel,
            "System State",
            "system_state"
        )
        
        # 3. Detection Active Status
        self.status_frames['detection_active'] = self._create_status_frame(
            right_panel,
            "Detection Active",
            "detection_active"
        )
        
        # 4. LiDAR Port Status
        self.status_frames['lidar_port_ok'] = self._create_status_frame(
            right_panel,
            "LiDAR Port",
            "lidar_port_ok"
        )
        
        # Start status update loop
        self._update_status_display()
        
    def _create_status_frame(self, parent, title, status_key):
        """
        Create a status LabelFrame
        
        Args:
            parent: Parent widget
            title: Title for the LabelFrame
            status_key: Key in status dictionary to display
            
        Returns:
            Dictionary with 'frame' and 'label' widgets
        """
        # Create LabelFrame
        frame = tk.LabelFrame(
            parent,
            text=title,
            font=('Arial', 10, 'bold'),
            bg='#95a5a6',
            fg='white',
            relief='ridge',
            bd=2
        )
        frame.pack(fill='x', pady=5)
        
        # Create value label
        label = tk.Label(
            frame,
            text='No Data',
            font=('Arial', 11),
            bg='#95a5a6',
            fg='white',
            height=2
        )
        label.pack(fill='both', padx=5, pady=5)
        
        return {
            'frame': frame,
            'label': label,
            'key': status_key
        }
    
    def _toggle_view_mode(self):
        """Toggle between RTSP and Map view modes"""
        if self.view_mode == "rtsp":
            self.view_mode = "map"
            self.view_mode_btn.config(text="View Mode: MAP")
            print("🗺️  Switched to MAP view")
        else:
            self.view_mode = "rtsp"
            self.view_mode_btn.config(text="View Mode: RTSP")
            print("🎥 Switched to RTSP view")
    
    def _update_status_display(self):
        """Update status indicators based on current status dictionary"""
        with self.status_lock:
            status = self.status_dict.copy()
        
        for status_frame in self.status_frames.values():
            frame = status_frame['frame']
            label = status_frame['label']
            key = status_frame['key']
            
            if key in status:
                value = status[key]
                
                # Handle boolean values
                if isinstance(value, bool):
                    if value:
                        frame.config(bg='#2ecc71')  # Green
                        label.config(bg='#2ecc71', text='✓ YES')
                    else:
                        frame.config(bg='#e74c3c')  # Red
                        label.config(bg='#e74c3c', text='✗ NO')
                
                # Handle string values
                elif isinstance(value, str):
                    if value.lower() in ['auto', 'active', 'ok', 'connected']:
                        frame.config(bg='#2ecc71')  # Green
                        label.config(bg='#2ecc71', text=value)
                    elif value.lower() in ['manual', 'inactive', 'error', 'disconnected']:
                        frame.config(bg='#f39c12')  # Orange
                        label.config(bg='#f39c12', text=f'⚠ {value}')
                    else:
                        frame.config(bg='#3498db')  # Blue for other states
                        label.config(bg='#3498db', text=value)
                
                # Handle other types
                else:
                    frame.config(bg='#3498db')  # Blue for other values
                    label.config(bg='#3498db', text=str(value))
                    
            else:
                # Key not in status dict
                frame.config(bg='#95a5a6')  # Gray
                label.config(bg='#95a5a6', text='No Data')
        
        # Schedule next update
        self.root.after(100, self._update_status_display)  # Update every 100ms
        
    def _setup_mqtt(self):
        """Setup MQTT client"""
        try:
            self.mqtt_client = mqtt.Client(client_id="operator_interface")
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            self.mqtt_client.on_message = self._on_mqtt_message
            
            print(f"🔗 Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}...")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            print(f"❌ Failed to setup MQTT: {e}")
            self.mqtt_connected = False
            
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            print("✅ Connected to MQTT broker")
            self.mqtt_connected = True
            self.status_label.config(text="MQTT: Connected", fg='#2ecc71')
            
            # Subscribe to status topic
            self.mqtt_client.subscribe(self.mqtt_status_topic)
            print(f"📡 Subscribed to status topic: {self.mqtt_status_topic}")
            
            # Subscribe to LiDAR scan topic
            self.mqtt_client.subscribe(self.mqtt_lidar_topic)
            print(f"📡 Subscribed to LiDAR topic: {self.mqtt_lidar_topic}")
        else:
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")
            self.mqtt_connected = False
            self.status_label.config(text="MQTT: Failed", fg='#e74c3c')
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        print("⚠️  Disconnected from MQTT broker")
        self.mqtt_connected = False
        self.status_label.config(text="MQTT: Disconnected", fg='#e74c3c')
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Callback for receiving MQTT messages"""
        try:
            # Check if this is a status message
            if msg.topic == self.mqtt_status_topic:
                # Parse JSON status dictionary
                status_json = msg.payload.decode('utf-8')
                status_dict = json.loads(status_json)
                
                # Update local status dictionary
                with self.status_lock:
                    self.status_dict = status_dict
            
            # NEW: Check if this is a LiDAR scan message
            elif msg.topic == self.mqtt_lidar_topic:
                # Parse JSON scan data
                scan_json = msg.payload.decode('utf-8')
                scan_dict = json.loads(scan_json)
                
                # Update latest LiDAR scan
                with self.lidar_scan_lock:
                    self.latest_lidar_scan = scan_dict
                
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON: {e}")
        except Exception as e:
            print(f"❌ Error processing MQTT message: {e}")
        
    def _send_command(self, command):
        """
        Send command via MQTT
        
        Args:
            command: Command string to send
        """
        if self.mqtt_connected:
            try:
                self.mqtt_client.publish(self.mqtt_topic, command)
                print(f"📤 Sent command: {command}")
            except Exception as e:
                print(f"❌ Failed to send command: {e}")
        else:
            print("⚠️  Cannot send command - MQTT not connected")
            
    def _on_key_press(self, event):
        """Handle keyboard arrow key presses"""
        key_map = {
            'Up': 'up',
            'Down': 'down',
            'Left': 'left',
            'Right': 'right'
        }
        
        if event.keysym in key_map:
            command = key_map[event.keysym]
            self._send_command(command)
    
    def _draw_map_view(self):
        """
        Generate map view image from LiDAR scan, static map points, and clusters
        
        Returns:
            numpy array: Image with map visualization
        """
        # Create white background
        image = np.ones((self.map_image_size, self.map_image_size, 3), dtype=np.uint8) * 255
        
        # Get status and scan data
        with self.status_lock:
            status = self.status_dict.copy()
        
        with self.lidar_scan_lock:
            lidar_scan = self.latest_lidar_scan
        
        # Draw grid (optional - uncomment if you want)
        # self._draw_grid(image)
        
        # LAYER 1: Draw static calibration map points (LIGHT GRAY - background)
        if 'static_map_points' in status and len(status['static_map_points']) > 0:
            for point in status['static_map_points']:
                x = point['x']
                y = point['y']
                
                # Convert to pixel coordinates
                px = int(self.map_center + x * self.map_scale)
                py = int(self.map_center - y * self.map_scale)
                
                # Draw if within bounds
                if 0 <= px < self.map_image_size and 0 <= py < self.map_image_size:
                    cv2.circle(image, (px, py), 10, (255, 0, 255), -1)  # Light gray
        
        # LAYER 2: Draw current LiDAR scan points (BLUE - live data)
        if lidar_scan and 'points' in lidar_scan:
            points = lidar_scan['points']
            for point in points:
                x = point['x']
                y = point['y']
                
                # Convert to pixel coordinates
                px = int(self.map_center + x * self.map_scale)
                py = int(self.map_center - y * self.map_scale)
                
                # Draw if within bounds
                if 0 <= px < self.map_image_size and 0 <= py < self.map_image_size:
                    cv2.circle(image, (px, py), 3, (255, 128, 0), -1)  # Orange/Blue
        
        # LAYER 3: Draw motion clusters (RED for closest, PURPLE for others)
        if 'clusters' in status and len(status['clusters']) > 0:
            for i, cluster in enumerate(status['clusters']):
                # First cluster (closest to origin) in RED, others in PURPLE
                if i == 0:
                    cluster_color = (0, 0, 255)  # Red in BGR
                    center_color = (0, 0, 200)
                else:
                    cluster_color = (255, 0, 255)  # Purple/Magenta in BGR
                    center_color = (200, 0, 200)
                
                # Draw cluster points
                if 'points' in cluster:
                    for point in cluster['points']:
                        x = point['x']
                        y = point['y']
                        
                        # Convert to pixel coordinates
                        px = int(self.map_center + x * self.map_scale)
                        py = int(self.map_center - y * self.map_scale)
                        
                        # Draw if within bounds
                        if 0 <= px < self.map_image_size and 0 <= py < self.map_image_size:
                            cv2.circle(image, (px, py), 5, cluster_color, -1)
                
                # Draw cluster center with larger circle
                if 'center' in cluster:
                    cx = cluster['center']['x']
                    cy = cluster['center']['y']
                    
                    # Convert to pixel coordinates
                    cpx = int(self.map_center + cx * self.map_scale)
                    cpy = int(self.map_center - cy * self.map_scale)
                    
                    # Draw if within bounds
                    if 0 <= cpx < self.map_image_size and 0 <= cpy < self.map_image_size:
                        # Draw center marker
                        cv2.circle(image, (cpx, cpy), 12, center_color, 2)
                        cv2.circle(image, (cpx, cpy), 4, center_color, -1)
                        
                        # Add label
                        label = f"C{i+1}"
                        cv2.putText(image, label, (cpx + 15, cpy - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, center_color, 2)
        
        # Draw origin marker (0, 0)
        cv2.drawMarker(image, (self.map_center, self.map_center), 
                      (0, 0, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(image, "Origin", (self.map_center + 15, self.map_center - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw scale reference
        scale_length = int(1.0 * self.map_scale)  # 1 meter
        cv2.line(image, (50, self.map_image_size - 50), 
                (50 + scale_length, self.map_image_size - 50), (0, 0, 0), 2)
        cv2.putText(image, "1m", (50, self.map_image_size - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(image, "Legend:", (10, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Static map
        cv2.circle(image, (20, legend_y + 25), 3, (200, 200, 200), -1)
        cv2.putText(image, "Static Map (Calibration)", (35, legend_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Current scan
        cv2.circle(image, (20, legend_y + 50), 3, (255, 128, 0), -1)
        cv2.putText(image, "Current Scan (Live)", (35, legend_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Cluster 1
        cv2.circle(image, (20, legend_y + 75), 5, (0, 0, 255), -1)
        cv2.putText(image, "Motion Cluster (Closest)", (35, legend_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Other clusters
        cv2.circle(image, (20, legend_y + 100), 5, (255, 0, 255), -1)
        cv2.putText(image, "Other Motion Clusters", (35, legend_y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Display scan info if available
        if lidar_scan:
            info_y = self.map_image_size - 80
            scan_num = lidar_scan.get('scan_number', 0)
            point_count = lidar_scan.get('point_count', 0)
            
            cv2.putText(image, f"Scan #{scan_num}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(image, f"Points: {point_count}", (10, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
            
    def _start_video_stream(self):
        """Start video stream capture"""
        if not self.rtsp_url:
            print("⚠️  No RTSP URL provided")
            return
            
        try:
            print(f"🎥 Connecting to video stream: {self.rtsp_url}")
            self.video_capture = cv2.VideoCapture(self.rtsp_url)
            
            if not self.video_capture.isOpened():
                print("❌ Failed to open video stream")
                return
                
            print("✅ Video stream connected")
            self.video_info_label.place_forget()  # Hide "waiting" message
            
            # Start video thread
            self.running = True
            video_thread = threading.Thread(target=self._video_capture_thread)
            video_thread.daemon = True
            video_thread.start()
            
            # Start display update
            self._update_video_display()
            
        except Exception as e:
            print(f"❌ Error starting video stream: {e}")
            
    def _video_capture_thread(self):
        """Background thread for capturing video frames"""
        while self.running and self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                print("⚠️  Failed to read frame")
                time.sleep(0.1)
                
    def _update_video_display(self):
        """Update the video/map display in the UI based on view mode"""
        if not self.running:
            return
        
        frame = None
        
        # Get frame based on view mode
        if self.view_mode == "rtsp":
            # Show RTSP video
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
        else:
            # Show map view
            map_image = self._draw_map_view()
            frame = cv2.cvtColor(map_image, cv2.COLOR_BGR2RGB)
        
        if frame is not None:
            # Resize to fit the display area
            height, width = frame.shape[:2]
            display_width = self.video_label.winfo_width()
            display_height = self.video_label.winfo_height()
            
            if display_width > 1 and display_height > 1:
                # Calculate aspect ratio
                aspect_ratio = width / height
                
                if display_width / display_height > aspect_ratio:
                    # Height is limiting factor
                    new_height = display_height
                    new_width = int(new_height * aspect_ratio)
                else:
                    # Width is limiting factor
                    new_width = display_width
                    new_height = int(new_width / aspect_ratio)
                
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert to PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update label
            self.video_label.config(image=photo)
            self.video_label.image = photo  # Keep a reference
                
        # Schedule next update
        self.root.after(30, self._update_video_display)  # ~30 FPS
        
    def _on_closing(self):
        """Handle window closing"""
        print("🛑 Shutting down...")
        self.running = False
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        if self.video_capture:
            self.video_capture.release()
            
        self.root.destroy()
        
    def run(self):
        """Start the operator interface"""
        # Start video stream if URL is provided
        if self.rtsp_url:
            self.root.after(1000, self._start_video_stream)  # Start after 1 second
            
        # Start Tkinter main loop
        self.root.mainloop()


def main():
    # Configuration
    MQTT_BROKER = '192.168.1.122'  # Change to your MQTT broker address
    MQTT_PORT = 1883
    MQTT_TOPIC = "camera/control"
    MQTT_STATUS_TOPIC = "camera/status"
    MQTT_LIDAR_TOPIC = "lidar/scan"  # NEW: LiDAR scan topic
    
    # RTSP URL (same as PersonAlarmManager uses)
    # Update with your camera's RTSP URL
    RTSP_URL = "rtsp://admin123:admin123@192.168.1.143:554/stream1"
    
    # Create and run operator interface
    operator = Operator(
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        mqtt_topic=MQTT_TOPIC,
        mqtt_status_topic=MQTT_STATUS_TOPIC,
        mqtt_lidar_topic=MQTT_LIDAR_TOPIC,  # NEW
        rtsp_url=RTSP_URL
    )
    
    operator.run()


if __name__ == "__main__":
    main()
