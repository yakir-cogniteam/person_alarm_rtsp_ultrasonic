#!/usr/bin/env python3

import cv2
import time
from onvif import ONVIFCamera
from zeep import wsse
import threading
import numpy as np


class PersonAlarmManager:
    def __init__(self, camera_ip, username, password, port=2020, pan_step=0.01, tilt_step=0.01, 
                 pan_speed=0.5, tilt_speed=0.5):
        """
        Initialize Person Alarm Manager for Tapo C200 camera
        
        Args:
            camera_ip (str): IP address of the camera
            username (str): Camera username
            password (str): Camera password
            port (int): ONVIF port (default 2020 for Tapo cameras)
            pan_step (float): Step size for pan adjustment (default 0.1)
            tilt_step (float): Step size for tilt adjustment (default 0.1)
            pan_speed (float): Pan movement speed in range [0.0, 1.0] (default 0.5)
            tilt_speed (float): Tilt movement speed in range [0.0, 1.0] (default 0.5)
        """
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
        self.running = False
        
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
            
            # Get stream URL (prioritize stream2 for lower latency)
            self._get_stream_url()
            
            # Initialize video capture with low-latency settings
            self._init_video_capture()
            
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
    
    def _execute_ptz_move(self, direction):
        """
        Execute PTZ move in a separate thread
        
        Args:
            direction (str): 'left', 'right', 'up', or 'down'
        """
        with self.ptz_lock:
            if direction == 'left':
                new_pan = self.current_pan - self.pan_step
                self.abs_pan(new_pan)
            elif direction == 'right':
                new_pan = self.current_pan + self.pan_step
                self.abs_pan(new_pan)
            elif direction == 'up':
                new_tilt = self.current_tilt + self.tilt_step
                self.abs_tilt(new_tilt)
            elif direction == 'down':
                new_tilt = self.current_tilt - self.tilt_step
                self.abs_tilt(new_tilt)
    
    def _handle_arrow_keys(self, key):
        """
        Handle arrow key presses for camera control
        
        Args:
            key: The key code from cv2.waitKey()
        """
        direction = None
        
        if key == 81 or key == 2:  # Left arrow
            direction = 'left'
            print(f"⬅️  Left arrow: pan {self.current_pan:.2f} -> {self.current_pan - self.pan_step:.2f} (speed: {self.pan_speed:.2f})")
            
        elif key == 83 or key == 3:  # Right arrow
            direction = 'right'
            print(f"➡️  Right arrow: pan {self.current_pan:.2f} -> {self.current_pan + self.pan_step:.2f} (speed: {self.pan_speed:.2f})")
            
        elif key == 82 or key == 0:  # Up arrow
            direction = 'up'
            print(f"⬆️  Up arrow: tilt {self.current_tilt:.2f} -> {self.current_tilt + self.tilt_step:.2f} (speed: {self.tilt_speed:.2f})")
            
        elif key == 84 or key == 1:  # Down arrow
            direction = 'down'
            print(f"⬇️  Down arrow: tilt {self.current_tilt:.2f} -> {self.current_tilt - self.tilt_step:.2f} (speed: {self.tilt_speed:.2f})")
        
        # Execute PTZ move in separate thread (non-blocking)
        if direction:
            if self.ptz_thread is None or not self.ptz_thread.is_alive():
                self.ptz_thread = threading.Thread(
                    target=self._execute_ptz_move,
                    args=(direction,)
                )
                self.ptz_thread.daemon = True
                self.ptz_thread.start()
    
    def run(self):
        """
        Main run loop - runs at 10Hz and displays video stream
        Press 'q' to quit
        """
        if not self.video_capture or not self.video_capture.isOpened():
            print("Video capture not initialized")
            return
        
        print("\n🎥 Starting Person Alarm Manager (Low-Latency Mode)...")
        print("=" * 50)
        print("Running at 10 Hz with background frame capture")
        print("Controls:")
        print("  Arrow Keys: Pan/Tilt camera (absolute positioning with speed)")
        print(f"  Pan step: {self.pan_step}, Pan speed: {self.pan_speed}")
        print(f"  Tilt step: {self.tilt_step}, Tilt speed: {self.tilt_speed}")
        print("  'q': Quit")
        print("=" * 50)
        
        self.running = True
        
        # Start background frame capture thread
        self.capture_thread = threading.Thread(target=self._frame_capture_thread)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        target_hz = 10
        target_interval = 1.0 / target_hz  # 0.1 seconds for 10 Hz
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
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
            
            # Add overlay information
            height, width = frame.shape[:2]
            
            # Add timestamp
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            # Add FPS counter
            if fps > 0:
                fps_text = f"FPS: {fps:.1f}"
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Add resolution and loop rate info
            info_text = f"{width}x{height} @ {target_hz}Hz (Low-Latency)"
            cv2.putText(frame, info_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, info_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Add PTZ position and speed info
            ptz_text = f"Pan: {self.current_pan:.2f} (speed: {self.pan_speed:.2f}) | Tilt: {self.current_tilt:.2f} (speed: {self.tilt_speed:.2f})"
            cv2.putText(frame, ptz_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, ptz_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show the frame
            cv2.imshow('Person Alarm Manager', frame)
            
            # Reduced waitKey for faster response (1ms instead of 30ms)
            key = cv2.waitKey(1) & 0xFF
            
            if key != 255:  # 255 means no key was pressed
                # Handle arrow keys (non-blocking)
                self._handle_arrow_keys(key)
            
            # Calculate sleep time to maintain 10 Hz
            loop_elapsed = time.time() - loop_start_time
            sleep_time = target_interval - loop_elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # If we're running slower than 10 Hz, print a warning occasionally
                if frame_count % 100 == 0:
                    print(f"⚠️ Warning: Loop running slower than {target_hz} Hz")
            
            frame_count += 1
        
        cv2.destroyAllWindows()
        print("Run loop stopped")
    
    def disconnect(self):
        """Clean up and disconnect"""
        self.running = False
        
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
    
    # Absolute positioning settings with speed control
    PAN_STEP = 0.1    # Step size for each arrow key press
    TILT_STEP = 0.1   # Step size for each arrow key press
    PAN_SPEED = 0.5   # Speed of movement (0.0 to 1.0) - higher is faster
    TILT_SPEED = 0.5  # Speed of movement (0.0 to 1.0) - higher is faster
    
    print("🎥 Person Alarm Manager - Tapo C200 (Low-Latency)")
    print("=" * 50)
    print(f"Camera IP: {CAMERA_IP}")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")
    print(f"Pan Step: {PAN_STEP}, Pan Speed: {PAN_SPEED}")
    print(f"Tilt Step: {TILT_STEP}, Tilt Speed: {TILT_SPEED}")
    print()
    
    # Create manager instance
    manager = PersonAlarmManager(
        CAMERA_IP, USERNAME, PASSWORD,
        pan_step=PAN_STEP,
        tilt_step=TILT_STEP,
        pan_speed=PAN_SPEED,
        tilt_speed=TILT_SPEED
    ) 
    
    try:
        # Connect to camera
        print("🔗 Connecting to camera...")
        if not manager.connect():
            print("\n❌ Failed to connect to camera. Please check:")
            print("1. Camera IP address is correct")
            print("2. Camera is powered on and connected to network")
            print("3. ONVIF is enabled in camera settings")
            print("4. Username and password are correct")
            return
        
        print("✅ Successfully connected!")
        
        # Start the main run loop
        print("\n▶️ Starting main run loop...")
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