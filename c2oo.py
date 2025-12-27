#!/usr/bin/env python3

import cv2
import time
from onvif import ONVIFCamera
from zeep import wsse
import threading
import numpy as np
import os
import pyaudio
import wave
import math
import paho.mqtt.client as mqtt
from scipy.spatial import KDTree


class PersonAlarmManager:
    def __init__(self, camera_ip, username, password, port=2020, pan_step=0.01, tilt_step=0.01, 
                 pan_speed=0.5, tilt_speed=0.5, enable_detection=True, detection_confidence=0.2,
                 mqtt_broker="localhost", mqtt_port=1883, mqtt_topic="camera/control", mqtt_status_topic="camera/status",
                 motion_threshold=0.5):

        self.ws_path = "/home/pi/person_alarm_ws/person_alarm_rtsp_ultrasonic"
        #self.ws_path = "/home/cogniteam-user/person_alarm_ws/person_alarm_rtsp_ultrasonic/"
        
        # MQTT settings
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.mqtt_status_topic = mqtt_status_topic
        self.mqtt_client = None
        self.mqtt_connected = False
        self.camera_ip = camera_ip
        self.username = username
        self.password = password
        self.port = port
        self.camera = None
        self.media_service = None
        self.ptz_service = None
        self.imaging_service = None
        self.stream_url = None
        self.video_capture = None
        self.lidar_port_ok = True
        self.running = False

        self.lidar_target_deg = None
        self.is_map_calibrated = False
        self.calibration_cmd = False
        self.is_calibration_active = False
        self.calibration_count = 0
        self.MAX_CALIBRATION_COUNT = 100      

        # NEW: KDTree and motion detection members
        self.calibration_points = []  # List to store 2D points during calibration
        self.kdtree = None  # KDTree structure for fast nearest neighbor queries
        self.motion_threshold = motion_threshold  # Distance threshold in meters (e.g., 0.5)
        self.motion_points = []  # List to store detected motion points in current frame      


        self.system_state = 'auto' # auto / manual
        self.rotating_to_target_active = False
        self.wanted_pan = None
        self.min_pan_deg = -180
        self.max_pan_deg = 180 
        self.conut_frame_for_detect = 0
        self.MAX_FRAMES_DETECTION = 20
        
        # Step sizes for arrow key adjustments
        self.pan_step = pan_step
        self.tilt_step = tilt_step
        
        # Speed settings for absolute moves
        self.pan_speed = pan_speed
        self.tilt_speed = tilt_speed
        
        # Current camera position
        self.current_pan = 0.0
        self.current_tilt = 0.0
        
        # Thread lock for PTZ commands
        self.ptz_lock = threading.Lock()
        
        # Track pending PTZ commands
        self.ptz_thread = None
        
        # Frame threading for reduced latency
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        self.frame_available = threading.Event()
        
        # Person detection settings
        self.enable_detection = enable_detection
        self.detection_confidence = detection_confidence
        self.net = None
        self.person_detected = False
        self.detection_count = 0
        self.last_detection_time = 0
        
        # MODIFIED: Detection activation control
        self.detection_active = False
        self.detection_start_time = 0
        self.detection_duration = 10.0  # 10 seconds
        
        # Audio alarm settings
        self.device_service = None
        self.audio_available = False
        self.beep_on_detection = True  # Enable/disable beep
        self.last_beep_time = 0
        self.beep_cooldown = 2.0  # Minimum seconds between beeps
        
        
        # Initialize detector if enabled
        if self.enable_detection:
            if(self._init_person_detector() == False):
                exit(-1)
    
    def disconnect(self):
        """Clean up and disconnect"""
        print("🛑 Stopping all threads...")
        self.running = False
        self.lidar_running = False  # Signal LIDAR thread to stop
        
        # Wait for LIDAR thread to stop
        if hasattr(self, 'lidar_thread') and self.lidar_thread and self.lidar_thread.is_alive():
            print("Waiting for LIDAR thread to stop...")
            self.lidar_thread.join(timeout=5.0)
        
        # ... rest of the disconnect code ...

    def _init_person_detector(self):
        """Initialize MobileNet SSD person detector (optimized for Raspberry Pi)"""
        try:
            print("Initializing person detector (MobileNet SSD)...")
            

            prototxt_path = self.ws_path + "/model/MobileNetSSD_deploy.prototxt"
            model_path = self.ws_path + "/model/MobileNetSSD_deploy.caffemodel"
           
            
            # Load the MobileNet SSD model
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            
            # Set backend to OpenCV for better Raspberry Pi compatibility
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            
            print("✅ Person detector initialized successfully")
            print(f"   Detection confidence threshold: {self.detection_confidence}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize person detector: {e}")
            print("   Person detection will be disabled")
            self.enable_detection = False
            return False
    
    def _setup_mqtt(self):
        """Setup MQTT client for receiving commands"""
        try:
            self.mqtt_client = mqtt.Client(client_id="camera_c200")
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
            # Subscribe to the control topic
            self.mqtt_client.subscribe(self.mqtt_topic)
            print(f"📡 Subscribed to topic: {self.mqtt_topic}")
        else:
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")
            self.mqtt_connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        print("⚠️  Disconnected from MQTT broker")
        self.mqtt_connected = False
    
    def _on_mqtt_message(self, client, userdata, msg):
        
        
        """Callback for receiving MQTT messages"""
        try:
            command = msg.payload.decode('utf-8')
            print(f"📥 Received command: {command}")

            self.current_pan, self.current_tilt, self.current_zoom = self.get_current_ptz()

            if command == 'left':
                new_pan = self.current_pan + self.pan_step
                self.abs_pan(new_pan)
            elif command == 'right':
                new_pan = self.current_pan - self.pan_step
                self.abs_pan(new_pan)
            elif command == 'up':
                new_tilt = self.current_tilt + self.tilt_step
                self.abs_tilt(new_tilt)
            elif command == 'down':
                new_tilt = self.current_tilt - self.tilt_step
                self.abs_tilt(new_tilt)
            elif command == 'go_home':
                self.go_home()
            elif command == 'switch_state':
                self.switch_state()
            elif command == 'sound_test':
                self.sound_test()
            elif command == 'calibrate':
                self.calibration_cmd = True
                
        except Exception as e:
            print(f"❌ Error processing MQTT message: {e}")
    
    
    def sound_test(self):

        self.play_beep()
    
    def switch_state(self):

        if self.system_state == 'manual':
            self.system_state = 'auto'
        elif self.system_state == 'auto':
            self.system_state = 'manual'
            

        print(f' the state now is {self.system_state}')    
    
    def _build_kdtree(self):
        """Build KDTree from collected calibration points"""
        if len(self.calibration_points) == 0:
            print("⚠️  No calibration points to build KDTree")
            return False
        
        try:
            # Convert list of points to numpy array
            points_array = np.array(self.calibration_points)
            self.kdtree = KDTree(points_array)
            print(f"✅ KDTree built with {len(self.calibration_points)} points")
            return True
        except Exception as e:
            print(f"❌ Failed to build KDTree: {e}")
            return False
    
    def get_distance_to_nearest_point(self, real_x, real_y):
        """
        Query the KDTree to find the distance to the nearest calibration point
        
        Args:
            real_x: X coordinate of the query point (meters)
            real_y: Y coordinate of the query point (meters)
            
        Returns:
            distance: Distance to nearest point (meters), or None if KDTree not available
        """
        if self.kdtree is None:
            return None
        
        try:
            query_point = np.array([real_x, real_y])
            distance, index = self.kdtree.query(query_point)
            return distance
        except Exception as e:
            print(f"❌ Error querying KDTree: {e}")
            return None
    
    def collect_motion_points(self, scan_points):
        """
        Collect all 2D points from scan that are above the motion threshold
        
        Args:
            scan_points: List of (real_x, real_y) tuples from the current scan
            
        Returns:
            motion_points: List of (real_x, real_y) tuples that exceed threshold
        """
        if not self.is_map_calibrated or self.kdtree is None:
            return []
        
        motion_points = []
        for real_x, real_y in scan_points:
            distance = self.get_distance_to_nearest_point(real_x, real_y)
            if distance is not None and distance > self.motion_threshold:
                motion_points.append((real_x, real_y))
        
        return motion_points
    
    def draw_motion_points(self, frame, motion_points, pixel_coords):
        """
        Draw motion points as red circles on the frame
        
        Args:
            frame: OpenCV image frame
            motion_points: List of (real_x, real_y) tuples representing motion
            pixel_coords: List of corresponding (pixel_x, pixel_y) tuples for drawing
        """
        for i, (real_x, real_y) in enumerate(motion_points):
            if i < len(pixel_coords):
                px, py = pixel_coords[i]
                # Draw red circle for motion point
                cv2.circle(frame, (int(px), int(py)), 5, (0, 0, 255), -1)
                # Optionally add a small label
                cv2.putText(frame, f"M", (int(px) + 8, int(py)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)    
    
    def play_beep(self):
        
        file_path = self.ws_path + "/sounds/beep.wav"
    
        try:
            # Open the WAV file
            wf = wave.open(file_path, 'rb')
            
            # Create PyAudio instance
            p = pyaudio.PyAudio()
            
            # Open stream
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(),
                            rate=wf.getframerate(),
                            output=True)
            
            # Read and play data in chunks
            chunk_size = 1024
            data = wf.readframes(chunk_size)
            
            print(f"🔊 Playing: {file_path}")
            
            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)
            
            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()
            
            print("✅ Playback finished")
            
        except FileNotFoundError:
            print(f"❌ Error: File '{file_path}' not found")
        except Exception as e:
            print(f"❌ Error playing audio: {e}")
       
    
    def _detect_persons(self, frame):
        """
        Detect persons in frame using MobileNet SSD
        
        Args:
            frame: Input frame (BGR image)
            
        Returns:
            detections: List of (confidence, x1, y1, x2, y2) tuples for detected persons
        """
        if not self.enable_detection or self.net is None:
            return []
        
        # MODIFIED: Only detect if detection is active
        if not self.detection_active:
            return []
        
        try:
            h, w = frame.shape[:2]
            
            # Resize for faster processing on Raspberry Pi
            # Use smaller input size for better performance
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)),
                0.007843,  # Scale factor
                (300, 300),
                127.5  # Mean subtraction
            )
            
            self.net.setInput(blob)
            detections = self.net.forward()
            
            persons = []
            
            # MobileNet SSD class IDs: 15 = person
            PERSON_CLASS_ID = 15
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                class_id = int(detections[0, 0, i, 1])
                
                # Check if it's a person with sufficient confidence
                if class_id == PERSON_CLASS_ID and confidence > self.detection_confidence:
                    # Get bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    persons.append((confidence, x1, y1, x2, y2))
            
            return persons
            
        except Exception as e:
            print(f"Error in person detection: {e}")
            return []
    
    def _draw_detections(self, frame, detections):
        """
        Draw detection boxes on frame
        
        Args:
            frame: Input frame
            detections: List of (confidence, x1, y1, x2, y2) tuples
            
        Returns:
            frame: Frame with drawn boxes
        """
        for conf, x1, y1, x2, y2 in detections:
            # Draw bounding box
            color = (0, 255, 0)  # Green for person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence label
            label = f"Person: {conf:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Background for label
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Text label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        
        return frame
        
    def connect(self):
        """Connect to the camera and initialize services"""
        try:
            print(f"Connecting to Tapo C200 at {self.camera_ip}:{self.port}")
            
            # Create ONVIF camera instance
            self.camera = ONVIFCamera(self.camera_ip, self.port, self.username, self.password)
            
            # Get media service
            self.media_service = self.camera.create_media_service()
            
            # Get PTZ service
            try:
                self.ptz_service = self.camera.create_ptz_service()
                print("PTZ service initialized successfully")
            except Exception as e:
                print(f"Warning: PTZ service not available: {e}")
            
            # Get imaging service
            try:
                self.imaging_service = self.camera.create_imaging_service()
                print("Imaging service initialized successfully")
            except Exception as e:
                print(f"Warning: Imaging service not available: {e}")
            
           
            
            # Initialize Tapo controller for alarm
            
            # Get stream URL (prioritize stream2 for lower latency)
            self._get_stream_url()
            
            # Initialize video capture with low-latency settings
            self._init_video_capture()
            
            # Setup MQTT server
            self._setup_mqtt()
            
            print("Successfully connected to Tapo C200")
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            return False
    
    def _get_stream_url(self):
        """Get the RTSP stream URL with priority on low-latency stream2"""
        # Prioritize stream2 (lower resolution, lower latency)
        working_urls = [
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/stream2",  # Lower latency
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/stream1",
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}/stream2",
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}/stream1"
        ]
        
        try:
            # First try to get the ONVIF stream URL
            profiles = self.media_service.GetProfiles()
            
            if profiles:
                # Try to find a lower resolution profile for lower latency
                for profile in profiles:
                    print(f"Available profile: {profile.Name}")
                
                profile = profiles[0]
                print(f"Using profile: {profile.Name}")
                
                stream_setup = self.media_service.create_type('GetStreamUri')
                stream_setup.ProfileToken = profile.token
                stream_setup.StreamSetup = {
                    'Stream': 'RTP-Unicast',
                    'Transport': {'Protocol': 'RTSP'}
                }
                
                stream_uri = self.media_service.GetStreamUri(stream_setup)
                onvif_url = stream_uri.Uri
                print(f"ONVIF Stream URL: {onvif_url}")
                
                if self._test_rtsp_url(onvif_url):
                    self.stream_url = onvif_url
                    print("Using ONVIF provided stream URL")
                    return
                
        except Exception as e:
            print(f"ONVIF stream URL failed: {e}")
        
        # Use the known working URLs (prioritizing stream2)
        print("Using tested working RTSP URL format (prioritizing stream2 for lower latency)...")
        for url in working_urls:
            print(f"Testing: {url}")
            if self._test_rtsp_url(url):
                self.stream_url = url
                print(f"Selected working stream URL: {url}")
                return
        
        # Fallback to stream2 (lower latency)
        self.stream_url = working_urls[0]
        print(f"Using fallback stream URL: {self.stream_url}")
    
    def _test_rtsp_url(self, url):
        """Test if an RTSP URL is accessible"""
        try:
            test_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            
            if test_cap.isOpened():
                result = [False, None]
                
                def read_frame():
                    ret, frame = test_cap.read()
                    result[0] = ret
                    result[1] = frame
                
                thread = threading.Thread(target=read_frame)
                thread.daemon = True
                thread.start()
                thread.join(timeout=3)
                
                test_cap.release()
                
                if thread.is_alive():
                    return False
                
                return result[0] and result[1] is not None
            
            test_cap.release()
            return False
            
        except Exception as e:
            print(f"URL test failed: {e}")
            return False
    
    def _init_video_capture(self):
        """Initialize video capture with low-latency settings"""
        try:
            print("Initializing video stream with low-latency settings...")
            print(f"Stream URL: {self.stream_url}")
            
            self.video_capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            
            # Critical: Set buffer size to 1 to minimize latency
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set additional low-latency properties
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    print(f"✅ Video stream initialized successfully!")
                    print(f"Frame size: {frame.shape}")
                    return True
                else:
                    print("❌ Could not read frame from stream")
                    return False
            else:
                print("❌ Failed to open video stream")
                return False
            
        except Exception as e:
            print(f"Failed to initialize video stream: {e}")
            return False
    
    def _lidar_thread(self):

        

        from pyrplidar import PyRPlidar
        import cv2
        import numpy as np
        import math
        import time

        # Initialize LIDAR
        lidar = PyRPlidar()
        lidar.connect(port="/dev/ttyUSB0", baudrate=115200, timeout=3)
        

        # Get device information
        try:
            info = lidar.get_info()
            print("info:", info)

            if info == 'device is connected':
                self.lidar_port_ok = True
        except Exception as e:
            self.lidar_port_ok = False

        health = lidar.get_health()
        print("health:", health)

        samplerate = lidar.get_samplerate()
        print("samplerate:", samplerate)

        # Start motor
        lidar.set_motor_pwm(500)

        # Get scan modes
        scan_modes = lidar.get_scan_modes()
        print("\nAvailable scan modes:")
        for idx, mode in enumerate(scan_modes):
            print(f"  Mode {idx}: {mode}")

        # Wait for motor to spin up
        time.sleep(2)

        # Visualization parameters
        IMAGE_SIZE = 800  # Size of the display window
        CENTER = IMAGE_SIZE // 2  # Center point of the image
        SCALE = 20  # Scale factor (pixels per mm) - adjusted for 3m range
        MAX_DISTANCE = 3500  # Maximum distance in mm to display

        # Start scanning with mode 2 (Boost)
        scan_mode = min(2, len(scan_modes) - 1)
        print(f"\nUsing scan mode: {scan_mode}")

        scan_generator = lidar.start_scan_express(scan_mode)()  # Call the function to get generator

        print("\nStarting visualization... Press 'q' to quit\n")

        frame_count = 0

        try:
            while True:
                # Create a white image
                image = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
                
               
                
                # Collect points for one complete scan (360 degrees)
                scan_points = []
                scan_started = False
                
                for measurement in scan_generator:
                    # PyRPlidarMeasurement has dict-like string representation
                    # Parse the measurement data
                    meas_str = str(measurement)
                    
                    # Extract values using string parsing (simple approach)
                    # Format: "{'start_flag': False, 'quality': 188, 'angle': 353.421875, 'distance': 1136.0}"
                    try:
                        # Simple parsing
                        start_flag = 'True' in meas_str.split("'start_flag': ")[1].split(',')[0]
                        quality = int(meas_str.split("'quality': ")[1].split(',')[0])
                        angle = float(meas_str.split("'angle': ")[1].split(',')[0])
                        distance = float(meas_str.split("'distance': ")[1].split('}')[0])
                    except:
                        continue  # Skip malformed data
                    
                    # If we see a start flag and we've already started collecting, we have a complete scan
                    if start_flag and scan_started:
                        break
                    
                    if start_flag:
                        scan_started = True
                    
                    # Collect valid points
                    if distance > 0 and distance < MAX_DISTANCE and quality > 0:
                        scan_points.append((angle, distance))
                    
                    # Safety: if we have too many points, break
                    if len(scan_points) > 10000:
                        break
                
                # Draw all collected points

                if  self.calibration_cmd == True:
                    self.calibration_cmd = False
                    self.is_map_calibrated = False
                    self.is_calibration_active = True
                    self.calibration_points = []  # Reset calibration points
                    self.calibration_count = 0

               
                # Collect 2D real-world coordinates and pixel coordinates
                scan_real_points = []  # (real_x, real_y) in mm
                scan_pixel_coords = []  # (px, py) for drawing

                valid_points = 0
                for angle, distance in scan_points:
                    # Convert polar coordinates to Cartesian
                    angle_rad = math.radians(angle)
                    
                   
                    # Calculate x, y coordinates (invert y for image coordinates)
                    x = int(CENTER + (distance / SCALE) * math.cos(angle_rad))
                    y = int(CENTER - (distance / SCALE) * math.sin(angle_rad))

                    real_x = float( (distance/1000.0) * math.cos(angle_rad)) 
                    real_y = float( (distance/1000.0) * math.sin(angle_rad)) 

                    # print(f' the distnace is {distance} real_x {real_x} real_y {real_y} ')
                    # Store real-world coordinates
                    scan_real_points.append((real_x, real_y))
                    
                    # Draw point if within image bounds
                    if 0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE:
                        scan_pixel_coords.append((x, y))
                        cv2.circle(image, (x, y), 2, (0, 0, 0), -1)
                        valid_points += 1
                    else:
                        scan_pixel_coords.append(None)  # Mark as out of bounds
                
                # CALIBRATION: Collect points during calibration phase
                if self.is_calibration_active :
                    for real_x, real_y in scan_real_points:
                        self.calibration_points.append([real_x, real_y])
                    
                    self.calibration_count += 1
                    
                    # Display calibration progress
                    calib_text = f"CALIBRATING: {self.calibration_count}/{self.MAX_CALIBRATION_COUNT}"
                    cv2.putText(image, calib_text, (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    if self.calibration_count >= self.MAX_CALIBRATION_COUNT:
                        self.is_calibration_active = False
                        self.is_map_calibrated = self._build_kdtree()
                        print(f"🎯 Calibration complete! Map calibrated: {self.is_map_calibrated}")
                
                # MOTION DETECTION: After calibration, detect motion points
                if self.is_map_calibrated and self.rotating_to_target_active == False and self.system_state == 'auto':
                    self.motion_points = self.collect_motion_points(scan_real_points)
                    
                    # Draw motion points as red circles
                    motion_count = 0
                    for i, (real_x, real_y) in enumerate(self.motion_points):
                        if i < len(scan_pixel_coords):
                            # Find corresponding pixel coordinate
                            for j, (rx, ry) in enumerate(scan_real_points):
                                if abs(rx - real_x) < 0.1 and abs(ry - real_y) < 0.1:
                                    if j < len(scan_pixel_coords) and scan_pixel_coords[j] is not None:
                                        px, py = scan_pixel_coords[j]
                                        cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
                                        motion_count += 1
                                    break
                    
                    # Display motion detection info
                    if motion_count > 0:
                        motion_text = f"MOTION: {motion_count} points"
                        cv2.putText(image, motion_text, (10, 120),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # If significant motion detected, trigger alarm logic
                        if motion_count > 10:  # Threshold for significant motion
                            # Find the angle of the centroid of motion points
                            motion_angles = []
                            for real_x, real_y in self.motion_points:
                                angle_to_point = math.degrees(math.atan2(real_y, real_x))
                                motion_angles.append(angle_to_point)
                            
                            if len(motion_angles) > 0:
                                avg_angle = sum(motion_angles) / len(motion_angles)
                                self.lidar_target_deg = avg_angle
                                print(f"🎯 Motion detected at angle: {avg_angle:.1f}°")
                
                # Add text information
                cv2.putText(image, f"Frame: {frame_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(image, f"Points: {valid_points}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                cv2.putText(image, "Press 'q' to quit", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # # Display the image
                # cv2.imshow('LIDAR Scan', image)
                
                # frame_count += 1
                
                # # Break loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

        except KeyboardInterrupt:
            self.lidar_port_ok = False

            print("\nStopping...")
        except Exception as e:
            self.lidar_port_ok = False

            print(f"\nError occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.lidar_port_ok = False

            # Cleanup
            print("Cleaning up...")
            lidar.set_motor_pwm(0)
            lidar.stop()
            lidar.disconnect()
            cv2.destroyAllWindows()
            print("LIDAR disconnected and cleanup complete")




    def _frame_capture_thread(self):
        """Continuously capture frames in background thread to avoid buffering"""
        print("Frame capture thread started")
        consecutive_failures = 0
        max_failures = 30
        
        while self.running:
            ret, frame = self.video_capture.read()
            
            if ret and frame is not None:
                consecutive_failures = 0
                with self.frame_lock:
                    self.latest_frame = frame
                self.frame_available.set()
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    print(f"❌ Frame capture failed {consecutive_failures} times, stopping...")
                    self.running = False
                    break
                time.sleep(0.01)  # Brief pause on failure
        
        print("Frame capture thread stopped")
    
    def abs_pan(self, pan_position, speed=None):
        """
        Send absolute pan command to the camera with speed
        
        Args:
            pan_position (float): Pan position in range [-1.0, 1.0]
                                 -1.0 = full left, 0.0 = center, 1.0 = full right
            speed (float): Pan speed in range [0.0, 1.0]. If None, uses self.pan_speed
        """
        if not self.ptz_service:
            print("PTZ service not available")
            return False
        
        try:
            # Clamp the value to valid range
            pan_position = max(-1.0, min(1.0, pan_position))
            
            # Use provided speed or default
            if speed is None:
                speed = self.pan_speed
            speed = max(0.0, min(1.0, speed))
            
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = profile.token
            request.Position = {
                'PanTilt': {'x': pan_position, 'y': self.current_tilt},
                'Zoom': {'x': 0.0}
            }
            # Set the speed for the movement
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
                'Zoom': 0.0
            }
            
            self.ptz_service.AbsoluteMove(request)
            self.current_pan = pan_position
            print(f"Absolute pan to position: {pan_position:.2f} at speed: {speed:.2f}")
            return True
            
        except Exception as e:
            print(f"Failed to execute absolute pan: {e}")
            return False
    
    def abs_tilt(self, tilt_position, speed=None):
        """
        Send absolute tilt command to the camera with speed
        
        Args:
            tilt_position (float): Tilt position in range [-1.0, 1.0]
                                  -1.0 = full down, 0.0 = center, 1.0 = full up
            speed (float): Tilt speed in range [0.0, 1.0]. If None, uses self.tilt_speed
        """
        if not self.ptz_service:
            print("PTZ service not available")
            return False
        
        try:
            # Clamp the value to valid range
            tilt_position = max(-1.0, min(1.0, tilt_position))
            
            # Use provided speed or default
            if speed is None:
                speed = self.tilt_speed
            speed = max(0.0, min(1.0, speed))
            
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = profile.token
            request.Position = {
                'PanTilt': {'x': self.current_pan, 'y': tilt_position},
                'Zoom': {'x': 0.0}
            }
            # Set the speed for the movement
            request.Speed = {
                'PanTilt': {'x': speed, 'y': speed},
                'Zoom': 0.0
            }
            
            self.ptz_service.AbsoluteMove(request)
            self.current_tilt = tilt_position
            print(f"Absolute tilt to position: {tilt_position:.2f} at speed: {speed:.2f}")
            return True
            
        except Exception as e:
            print(f"Failed to execute absolute tilt: {e}")
            return False
    
    # def _execute_ptz_move(self, direction):
    #     """
    #     Execute PTZ move in a separate thread
        
    #     Args:
    #         direction (str): 'left', 'right', 'up', 'down', or 'home'
    #     """
    #     with self.ptz_lock:
    #         if direction == 'left':
    #             new_pan = self.current_pan - self.pan_step
    #             self.abs_pan(new_pan)
    #         elif direction == 'right':
    #             new_pan = self.current_pan + self.pan_step
    #             self.abs_pan(new_pan)
    #         elif direction == 'up':
    #             new_tilt = self.current_tilt + self.tilt_step
    #             self.abs_tilt(new_tilt)
    #         elif direction == 'down':
    #             new_tilt = self.current_tilt - self.tilt_step
    #             self.abs_tilt(new_tilt)
    #         elif direction == 'home':
    #             self.go_home()
    
    def go_home(self):
        print('go home')
        self.abs_pan(0.0, 1)
        self.abs_tilt(0.0, 1)


    def _handle_arrow_keys(self, key):
        """
        Handle arrow key presses for camera control
        
        Args:
            key: The key code from cv2.waitKey()
        """
        direction = None
        
        if key == 83 or key == 3:  # Left arrow
            direction = 'left'
            print(f"⬅️  Left arrow: pan {self.current_pan:.2f} -> {self.current_pan - self.pan_step:.2f} (speed: {self.pan_speed:.2f})")
            
        elif key == 81 or key == 2:  # Right arrow
            direction = 'right'
            print(f"➡️  Right arrow: pan {self.current_pan:.2f} -> {self.current_pan + self.pan_step:.2f} (speed: {self.pan_speed:.2f})")
            
        elif key == 82 or key == 0:  # Up arrow
            direction = 'up'
            print(f"⬆️  Up arrow: tilt {self.current_tilt:.2f} -> {self.current_tilt + self.tilt_step:.2f} (speed: {self.tilt_speed:.2f})")
            
        elif key == 84 or key == 1:  # Down arrow
            direction = 'down'
            print(f"⬇️  Down arrow: tilt {self.current_tilt:.2f} -> {self.current_tilt - self.tilt_step:.2f} (speed: {self.tilt_speed:.2f})")
        
        elif key == ord('h') or key == ord('H'):  # 'h' or 'H' key

            direction = 'home'

        elif key == ord('b') or key == ord('B'):  # BEEP Key:
            self.play_beep()

        # Execute PTZ move in separate thread (non-blocking)
        if direction:
            if self.ptz_thread is None or not self.ptz_thread.is_alive():
                self.ptz_thread = threading.Thread(
                    target=self._execute_ptz_move,
                    args=(direction,)
                )
                self.ptz_thread.daemon = True
                self.ptz_thread.start()
    
   
    
    def rotate_to_target(self, lidar_target_deg):
        onvif_pan = self.pan_degrees_to_onvif(lidar_target_deg)
        self.abs_pan(onvif_pan, 1)

        self.wanted_pan = onvif_pan


    def pan_degrees_to_onvif(self, degrees):
        """Convert degrees to ONVIF normalized value"""
        # Clamp to valid range first
        deg_clamped = max(self.min_pan_deg, min(self.max_pan_deg, degrees))
        
        # Map degree range to ONVIF range [-1, 1]
        onvif_value = ((deg_clamped - self.min_pan_deg) * 2.0 / (self.max_pan_deg - self.min_pan_deg)) - 1.0
        
        return onvif_value * -1.0
    
    def pan_onvif_to_degrees(self, onvif_value):
        """Convert ONVIF normalized value to degrees"""
        # Clamp ONVIF value to valid range
        onvif_clamped = max(-1.0, min(1.0, onvif_value))
        
        # Map ONVIF range [-1, 1] to degree range
        degrees = -1.0 * ((onvif_clamped + 1.0) * (self.max_pan_deg - self.min_pan_deg) / 2.0) + self.min_pan_deg
        
        return degrees

    def get_current_ptz(self):
        try:
            # Get first available profile
            media_service = self.camera.create_media_service()
            profiles = media_service.GetProfiles()
            profile_token = profiles[0].token if profiles else None
            
            if not profile_token:
                print("No media profiles available")
                return 0.0, 0.0, 0.0
            
            status = self.ptz_service.GetStatus({'ProfileToken': profile_token})
            
            # Check if status is valid
            if status is None:
                print("Status is None")
                return 0.0, 0.0, 0.0
            
            # Access attributes directly (not using .get())
            if not hasattr(status, 'Position') or status.Position is None:
                print("Position attribute not available")
                return 0.0, 0.0, 0.0
            
            pos = status.Position
            
            # Access PanTilt and Zoom attributes directly
            pan = pos.PanTilt.x if hasattr(pos, 'PanTilt') and pos.PanTilt else 0.0
            tilt = pos.PanTilt.y if hasattr(pos, 'PanTilt') and pos.PanTilt else 0.0
            zoom = pos.Zoom.x if hasattr(pos, 'Zoom') and pos.Zoom else 0.0

            return pan, tilt, zoom
            
        except Exception as e:
            print(f'Error getting PTZ: {e}')
            import traceback
            traceback.print_exc()
            return 0.0, 0.0, 0.0

    def publish_status(self):
        """Publish current system status to MQTT"""
        if not self.mqtt_connected or not self.mqtt_client:
            return
        
        try:
            import json
            
            # Create status dictionary
            status = {
                "is_map_calibrated": self.is_map_calibrated,
                "system_state": self.system_state,
                "detection_active": self.detection_active,
                "lidar_port_ok": self.lidar_port_ok,
                "rotating_to_target": self.rotating_to_target_active,
                "current_pan": self.current_pan,
                "current_tilt": self.current_tilt,
                "timestamp": time.time()
            }
            
            # Convert to JSON and publish
            status_json = json.dumps(status)
            self.mqtt_client.publish(self.mqtt_status_topic, status_json)
            
        except Exception as e:
            print(f"❌ Error publishing status: {e}")

    def run(self):
     
        if not self.video_capture or not self.video_capture.isOpened():
            print("Video capture not initialized")
            return
   
        self.go_home()
        self.running = True
        
        # Start background frame capture thread
        self.capture_thread = threading.Thread(target=self._frame_capture_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start background frame capture thread
        self.lidar_thread = threading.Thread(target=self._lidar_thread)
        self.lidar_thread.daemon = True
        self.lidar_thread.start()
        
        target_hz = 10
        target_interval = 1.0 / target_hz  # 0.1 seconds for 10 Hz
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        # Detection display toggle
        show_detections = True
        
        # Detection timing (run detection less frequently for performance)
        detection_interval = 0.2  # Run detection every 200ms (5 Hz)
        last_detection_run = 0
        cached_detections = []
        
        # Status publishing timing
        status_publish_interval = 0.5  # Publish status every 500ms (2 Hz)
        last_status_publish = 0
        
        # Wait for first frame
        if not self.frame_available.wait(timeout=5.0):
            print("❌ Timeout waiting for first frame")
            self.running = False
            return
        
        while self.running:
            loop_start_time = time.time()
            
            # Get latest frame from background thread
            with self.frame_lock:
                if self.latest_frame is None:
                    time.sleep(0.001)
                    continue
                frame = self.latest_frame.copy()
            
            fps_counter += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_counter = 0
            

            if self.rotating_to_target_active:
                self.lidar_target_deg = None

                pan, tilt, zoom = self.get_current_ptz()
                
                if math.fabs(self.wanted_pan - pan) < 0.05:

                    self.conut_frame_for_detect+= 1


                    if self.conut_frame_for_detect > self.MAX_FRAMES_DETECTION:        
                        self.rotating_to_target_active = False
                        self.conut_frame_for_detect = 0

                    self.enable_detection = True
                    self.detection_active = True
                    detections = self._detect_persons(frame)
                    if len(detections) > 0:

                        self.rotating_to_target_active = False
                        self.conut_frame_for_detect = 0
                        
                        print(f"🚨 PERSON DETECTED! (Count: {self.detection_count})")
                            
                        self.play_beep()
                        time.sleep(0.1)
                        self.play_beep()
                        time.sleep(0.1)
                        self.play_beep()


                        self.enable_detection = False
                        self.detection_active = False

                        self.go_home()

                    else:
                        self.rotating_to_target_active = False    
                        print(' noooo person no person !!!!!') 
                        self.go_home() 
                
            elif self.lidar_target_deg != None:

                self.rotate_to_target(self.lidar_target_deg)
                self.lidar_target_deg = None
                self.rotating_to_target_active = True
                

            if self.system_state == 'manual' and self.rotating_to_target_active:
                self.enable_detection = False
                self.detection_active = False
                self.rotating_to_target_active = False   

                self.go_home()

          


            
            # Add overlay information
            height, width = frame.shape[:2]
            
            
            # Add FPS counter
            if fps > 0:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
       
            # # Show the frame
            # cv2.imshow('Person Alarm Manager', frame)
            
            # # Reduced waitKey for faster response (1ms instead of 30ms)
            # key = cv2.waitKey(1) & 0xFF
            
            # if key == ord('q'):
            #     print("\n⏹️  Quit key pressed")
            #     break            
            # elif key != 255:  # 255 means no key was pressed
            #     # Handle arrow keys (non-blocking)
            #     self._handle_arrow_keys(key)
            
            # Publish status periodically
            if current_time - last_status_publish >= status_publish_interval:
                self.publish_status()
                last_status_publish = current_time
            
            # Calculate sleep time to maintain 10 Hz
            loop_elapsed = time.time() - loop_start_time
            sleep_time = target_interval - loop_elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            
            frame_count += 1
        
        cv2.destroyAllWindows()
        print("Run loop stopped")

    def disconnect(self):
        """Clean up and disconnect"""
        self.running = False
        
        # Stop MQTT client
        if self.mqtt_client:
            print("Disconnecting MQTT client...")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        
        # Wait for capture thread to stop
        if self.capture_thread and self.capture_thread.is_alive():
            print("Waiting for capture thread to stop...")
            self.capture_thread.join(timeout=2.0)
        
        # Wait for any pending PTZ commands to complete
        if self.ptz_thread and self.ptz_thread.is_alive():
            print("Waiting for PTZ command to complete...")
            self.ptz_thread.join(timeout=2.0)
        
        if self.video_capture:
            self.video_capture.release()
        
        cv2.destroyAllWindows()
        print("Disconnected from camera")


def main():
    # Camera configuration - Update these values
    CAMERA_IP = "192.168.1.143"
    USERNAME = "admin123"
    PASSWORD = "admin123"
    
    # MQTT configuration
    MQTT_BROKER = "localhost"  # Change to your MQTT broker address
    MQTT_PORT = 1883
    MQTT_TOPIC = "camera/control"
    MQTT_STATUS_TOPIC = "camera/status"  # Status publishing topic
    
    # Absolute positioning settings with speed control
    PAN_STEP = 0.1    # Step size for each arrow key press
    TILT_STEP = 0.1   # Step size for each arrow key press
    PAN_SPEED = 0.5   # Speed of movement (0.0 to 1.0) - higher is faster
    TILT_SPEED = 0.5  # Speed of movement (0.0 to 1.0) - higher is faster
    
    # Person detection settings
    ENABLE_DETECTION = True      # Set to False to disable person detection
    DETECTION_CONFIDENCE = 0.5   # Confidence threshold (0.0 to 1.0)
    
    # Motion detection settings
    MOTION_THRESHOLD = 0.5 # Distance in meters (500mm) to consider as motion
 
    # Create manager instance
    manager = PersonAlarmManager(
        CAMERA_IP, USERNAME, PASSWORD,
        pan_step=PAN_STEP,
        tilt_step=TILT_STEP,
        pan_speed=PAN_SPEED,
        tilt_speed=TILT_SPEED,
        enable_detection=ENABLE_DETECTION,
        detection_confidence=DETECTION_CONFIDENCE,
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        mqtt_topic=MQTT_TOPIC,
        mqtt_status_topic=MQTT_STATUS_TOPIC,
        motion_threshold=MOTION_THRESHOLD
    ) 
    
    try:
        # Connect to camera
        print("🔗 Connecting to camera...")
        if not manager.connect():       
            return
        
        print("✅ Successfully connected!")
        
        # Start the main run loop
        manager.run()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        manager.disconnect()
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()