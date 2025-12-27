#!/usr/bin/env python3

import tkinter as tk
from tkinter import ttk
import paho.mqtt.client as mqtt
import threading
import json
import numpy as np
import cv2
from PIL import Image, ImageTk
import time
import math


class LidarVisualizer:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883, 
                 mqtt_topic="lidar/scan"):
        """
        Initialize LiDAR visualizer
        
        Args:
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            mqtt_topic: Topic to subscribe for scan data
        """
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # Scan data
        self.latest_scan = None
        self.scan_lock = threading.Lock()
        
        # Visualization parameters
        self.image_size = 800  # Size of the display window
        self.center = self.image_size // 2  # Center point
        self.scale = 100  # Pixels per meter (adjustable)
        self.max_distance = 4.0  # Maximum distance in meters to display
        
        # Statistics
        self.scans_received = 0
        self.last_scan_time = 0
        self.scan_rate = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.display_fps = 0
        
        # Display options
        self.show_grid = True
        self.show_origin = True
        self.show_scale = True
        self.point_size = 2
        self.point_color = (0, 0, 255)  # Red in BGR
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("LiDAR Scan Visualizer")
        self.root.geometry("1000x850")
        
        # Setup UI
        self._setup_ui()
        
        # Setup MQTT
        self._setup_mqtt()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_ui(self):
        """Setup the Tkinter user interface"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2c3e50')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="LiDAR Scan Visualizer",
            font=('Arial', 16, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=(0, 10))
        
        # Top panel for controls
        control_panel = tk.Frame(main_frame, bg='#34495e', relief='ridge', bd=2)
        control_panel.pack(fill='x', padx=5, pady=5)
        
        # Left side - Statistics
        stats_frame = tk.Frame(control_panel, bg='#34495e')
        stats_frame.pack(side='left', padx=10, pady=5)
        
        self.mqtt_status_label = tk.Label(
            stats_frame,
            text="MQTT: Disconnected",
            font=('Arial', 10),
            bg='#34495e',
            fg='#e74c3c'
        )
        self.mqtt_status_label.pack(anchor='w')
        
        self.scan_rate_label = tk.Label(
            stats_frame,
            text="Scan Rate: 0.0 Hz",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        )
        self.scan_rate_label.pack(anchor='w')
        
        self.point_count_label = tk.Label(
            stats_frame,
            text="Points: 0",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        )
        self.point_count_label.pack(anchor='w')
        
        self.display_fps_label = tk.Label(
            stats_frame,
            text="Display FPS: 0.0",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        )
        self.display_fps_label.pack(anchor='w')
        
        # Right side - Controls
        controls_frame = tk.Frame(control_panel, bg='#34495e')
        controls_frame.pack(side='right', padx=10, pady=5)
        
        # Scale control
        scale_frame = tk.Frame(controls_frame, bg='#34495e')
        scale_frame.pack(side='left', padx=10)
        
        tk.Label(
            scale_frame,
            text="Zoom:",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        ).pack(side='left')
        
        self.scale_var = tk.IntVar(value=self.scale)
        scale_slider = tk.Scale(
            scale_frame,
            from_=20,
            to=200,
            orient='horizontal',
            variable=self.scale_var,
            command=self._on_scale_change,
            bg='#34495e',
            fg='white',
            highlightthickness=0,
            length=150
        )
        scale_slider.pack(side='left', padx=5)
        
        # Point size control
        point_frame = tk.Frame(controls_frame, bg='#34495e')
        point_frame.pack(side='left', padx=10)
        
        tk.Label(
            point_frame,
            text="Point Size:",
            font=('Arial', 10),
            bg='#34495e',
            fg='white'
        ).pack(side='left')
        
        self.point_size_var = tk.IntVar(value=self.point_size)
        point_slider = tk.Scale(
            point_frame,
            from_=1,
            to=10,
            orient='horizontal',
            variable=self.point_size_var,
            command=self._on_point_size_change,
            bg='#34495e',
            fg='white',
            highlightthickness=0,
            length=100
        )
        point_slider.pack(side='left', padx=5)
        
        # Display canvas
        canvas_frame = tk.Frame(main_frame, bg='black', relief='sunken', bd=2)
        canvas_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        self.canvas_label = tk.Label(canvas_frame, bg='black')
        self.canvas_label.pack(fill='both', expand=True)
        
        # Info label (shown when no data)
        self.info_label = tk.Label(
            canvas_frame,
            text="Waiting for LiDAR data...",
            font=('Arial', 14),
            bg='black',
            fg='white'
        )
        self.info_label.place(relx=0.5, rely=0.5, anchor='center')
        
    def _on_scale_change(self, value):
        """Handle scale slider change"""
        self.scale = int(value)
        
    def _on_point_size_change(self, value):
        """Handle point size slider change"""
        self.point_size = int(value)
        
    def _setup_mqtt(self):
        """Setup MQTT client"""
        try:
            self.mqtt_client = mqtt.Client(client_id="lidar_visualizer")
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
            self.mqtt_status_label.config(text="MQTT: Connected", fg='#2ecc71')
            
            # Subscribe to scan topic
            self.mqtt_client.subscribe(self.mqtt_topic)
            print(f"📡 Subscribed to topic: {self.mqtt_topic}")
        else:
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")
            self.mqtt_connected = False
            self.mqtt_status_label.config(text="MQTT: Failed", fg='#e74c3c')
            
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        print("⚠️  Disconnected from MQTT broker")
        self.mqtt_connected = False
        self.mqtt_status_label.config(text="MQTT: Disconnected", fg='#e74c3c')
    
    def _on_mqtt_message(self, client, userdata, msg):
        """Callback for receiving MQTT messages"""
        try:
            # Parse JSON scan data
            scan_json = msg.payload.decode('utf-8')
            scan_data = json.loads(scan_json)
            
            # Update latest scan
            with self.scan_lock:
                self.latest_scan = scan_data
                self.scans_received += 1
                
                # Calculate scan rate
                current_time = time.time()
                if self.last_scan_time > 0:
                    delta = current_time - self.last_scan_time
                    if delta > 0:
                        self.scan_rate = 0.9 * self.scan_rate + 0.1 * (1.0 / delta)  # Smoothed
                self.last_scan_time = current_time
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse scan JSON: {e}")
        except Exception as e:
            print(f"❌ Error processing MQTT message: {e}")
    
    def _draw_grid(self, image):
        """Draw grid lines on the image"""
        if not self.show_grid:
            return
        
        grid_color = (200, 200, 200)  # Light gray
        
        # Draw circles at 1m, 2m, 3m intervals
        for radius_m in range(1, int(self.max_distance) + 1):
            radius_px = int(radius_m * self.scale)
            cv2.circle(image, (self.center, self.center), radius_px, grid_color, 1)
            
            # Label the circle
            label = f"{radius_m}m"
            cv2.putText(image, label, (self.center + 5, self.center - radius_px + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
        
        # Draw angle lines (every 30 degrees)
        for angle_deg in range(0, 360, 30):
            angle_rad = math.radians(angle_deg)
            end_x = int(self.center + self.max_distance * self.scale * math.cos(angle_rad))
            end_y = int(self.center - self.max_distance * self.scale * math.sin(angle_rad))
            cv2.line(image, (self.center, self.center), (end_x, end_y), grid_color, 1)
            
            # Label the angle
            label_dist = (self.max_distance - 0.3) * self.scale
            label_x = int(self.center + label_dist * math.cos(angle_rad))
            label_y = int(self.center - label_dist * math.sin(angle_rad))
            cv2.putText(image, f"{angle_deg}°", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, grid_color, 1)
    
    def _draw_origin(self, image):
        """Draw origin marker"""
        if not self.show_origin:
            return
        
        origin_color = (0, 255, 0)  # Green
        cv2.drawMarker(image, (self.center, self.center), origin_color,
                      cv2.MARKER_CROSS, 20, 2)
        cv2.putText(image, "Origin", (self.center + 15, self.center - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, origin_color, 2)
    
    def _draw_scale_reference(self, image):
        """Draw scale reference"""
        if not self.show_scale:
            return
        
        scale_color = (255, 255, 255)  # White
        scale_length = int(1.0 * self.scale)  # 1 meter
        
        start_x = 50
        start_y = self.image_size - 50
        
        cv2.line(image, (start_x, start_y), (start_x + scale_length, start_y),
                scale_color, 2)
        cv2.line(image, (start_x, start_y - 5), (start_x, start_y + 5),
                scale_color, 2)
        cv2.line(image, (start_x + scale_length, start_y - 5),
                (start_x + scale_length, start_y + 5), scale_color, 2)
        
        cv2.putText(image, "1m", (start_x + scale_length // 2 - 15, start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, scale_color, 2)
    
    def _draw_scan(self):
        """Draw the LiDAR scan"""
        # Create white background
        image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255
        
        # Draw grid
        self._draw_grid(image)
        
        # Get latest scan data
        with self.scan_lock:
            scan = self.latest_scan
        
        if scan is None:
            return image
        
        # Hide info label once we have data
        if self.info_label.winfo_viewable():
            self.info_label.place_forget()
        
        # Draw scan points
        points = scan.get('points', [])
        valid_points = 0
        
        for point in points:
            x = point['x']  # meters
            y = point['y']  # meters
            
            # Convert to pixel coordinates
            px = int(self.center + x * self.scale)
            py = int(self.center - y * self.scale)  # Invert Y for image coordinates
            
            # Draw if within bounds
            if 0 <= px < self.image_size and 0 <= py < self.image_size:
                cv2.circle(image, (px, py), self.point_size, self.point_color, -1)
                valid_points += 1
        
        # Draw origin
        self._draw_origin(image)
        
        # Draw scale reference
        self._draw_scale_reference(image)
        
        # Draw statistics on image
        stats_y = 30
        cv2.putText(image, f"Points: {len(points)} ({valid_points} visible)",
                   (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(image, f"Scan Rate: {self.scan_rate:.1f} Hz",
                   (10, stats_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(image, f"Scan #: {scan.get('scan_number', 0)}",
                   (10, stats_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(image, f"Display FPS: {self.display_fps:.1f}",
                   (10, stats_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    def _update_display(self):
        """Update the display with latest scan"""
        # Draw scan
        image = self._draw_scan()
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        pil_image = Image.fromarray(image_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update canvas
        self.canvas_label.config(image=photo)
        self.canvas_label.image = photo  # Keep reference
        
        # Update statistics labels
        with self.scan_lock:
            scan = self.latest_scan
            
        if scan:
            point_count = len(scan.get('points', []))
            self.point_count_label.config(text=f"Points: {point_count}")
            self.scan_rate_label.config(text=f"Scan Rate: {self.scan_rate:.1f} Hz")
        
        # Calculate display FPS
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            self.display_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.display_fps_label.config(text=f"Display FPS: {self.display_fps:.1f}")
            self.fps_counter = 0
            self.fps_start_time = current_time
        
        # Schedule next update (30 FPS)
        self.root.after(33, self._update_display)
    
    def _on_closing(self):
        """Handle window closing"""
        print("🛑 Shutting down...")
        
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            
        self.root.destroy()
        
    def run(self):
        """Start the visualizer"""
        # Start display update loop
        self.root.after(100, self._update_display)
        
        # Start Tkinter main loop
        self.root.mainloop()


def main():
    # Configuration
    MQTT_BROKER = "192.168.1.122"  # Change to your Pi's IP address
    MQTT_PORT = 1883
    MQTT_TOPIC = "lidar/scan"
    
    print("🎯 LiDAR Scan Visualizer")
    print("=" * 50)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print(f"MQTT Topic: {MQTT_TOPIC}")
    print("=" * 50)
    
    # Create and run visualizer
    visualizer = LidarVisualizer(
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        mqtt_topic=MQTT_TOPIC
    )
    
    visualizer.run()


if __name__ == "__main__":
    main()