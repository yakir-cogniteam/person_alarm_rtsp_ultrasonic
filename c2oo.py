#!/usr/bin/env python3

# sudo apt install python3-opencv
# pip3 install onvif-zeep --break-system-packages
# cd /home/pi/.local/lib/python3.11/site-packages/wsdl/
# cd /home/pi/.local/lib/python3.11/site-packages/wsdl
# cd /home/pi/.local/lib/python3.11/site-packages/
# mkdir -p wsdl
# cd wsdl/
# wget https://www.onvif.org/ver10/device/wsdl/devicemgmt.wsdl
# wget https://www.onvif.org/ver10/media/wsdl/media.wsdl
# wget https://www.onvif.org/ver20/ptz/wsdl/ptz.wsdl
# wget https://www.onvif.org/ver10/events/wsdl/event.wsdl
# wget https://www.onvif.org/ver20/imaging/wsdl/imaging.wsdl

# cd /home/pi/.local/lib/
# mkdir -p ver10/schema
# mkdir -p ver20/schema
# wget https://www.onvif.org/ver10/schema/onvif.xsd
# wget https://www.onvif.org/ver10/schema/common.xsd
# cd ../../ver20/schema
# wget https://www.onvif.org/ver20/schema/onvif.xsd


import cv2
import time
from onvif import ONVIFCamera
from zeep import wsse
import threading
import numpy as np
import requests
from urllib.parse import urlparse

class TapoC200Controller:
    def __init__(self, camera_ip, username, password, port=2020):
        """
        Initialize Tapo C200 camera connection
        
        Args:
            camera_ip (str): IP address of the camera
            username (str): Camera username (usually 'admin')
            password (str): Camera password
            port (int): ONVIF port (default 2020 for Tapo cameras)
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
            
            # Get stream URL
            self._get_stream_url()
            
            print("Successfully connected to Tapo C200")
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            return False
    
    def _get_stream_url(self):
        """Get the RTSP stream URL with multiple fallback options"""
        # Based on the test results, we know these URLs work:
        working_urls = [
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/stream1",  # High quality
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}:554/stream2",  # Lower quality
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}/stream1",
            f"rtsp://{self.username}:{self.password}@{self.camera_ip}/stream2"
        ]
        
        try:
            # First try to get the ONVIF stream URL
            profiles = self.media_service.GetProfiles()
            
            if profiles:
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
                
                # Test if ONVIF URL works
                if self._test_rtsp_url(onvif_url):
                    self.stream_url = onvif_url
                    print("Using ONVIF provided stream URL")
                    return
                
        except Exception as e:
            print(f"ONVIF stream URL failed: {e}")
        
        # Use the known working URLs
        print("Using tested working RTSP URL format...")
        for url in working_urls:
            print(f"Testing: {url}")
            if self._test_rtsp_url(url):
                self.stream_url = url
                print(f"Selected working stream URL: {url}")
                return
        
        # Fallback to first known working format
        self.stream_url = working_urls[0]
        print(f"Using fallback stream URL: {self.stream_url}")
    
    def _test_rtsp_url(self, url):
        """Test if an RTSP URL is accessible"""
        try:
            # Quick test with OpenCV
            test_cap = cv2.VideoCapture(url)
            
            # Try to set timeout if available (newer OpenCV versions)
            try:
                test_cap.set(cv2.CAP_PROP_TIMEOUT, 5000)  # 5 second timeout
            except AttributeError:
                # Older OpenCV versions don't have CAP_PROP_TIMEOUT
                pass
            
            if test_cap.isOpened():
                # Use a simple timeout mechanism for older OpenCV
                import threading
                import time
                
                result = [False, None]
                
                def read_frame():
                    ret, frame = test_cap.read()
                    result[0] = ret
                    result[1] = frame
                
                thread = threading.Thread(target=read_frame)
                thread.daemon = True
                thread.start()
                thread.join(timeout=3)  # 3 second timeout
                
                test_cap.release()
                
                if thread.is_alive():
                    print(f"Timeout testing URL: {url}")
                    return False
                
                return result[0] and result[1] is not None
            
            test_cap.release()
            return False
            
        except Exception as e:
            print(f"URL test failed: {e}")
            return False
    
    def start_video_stream(self):
        """Start video streaming with enhanced error handling"""
        try:
            print("Starting video stream...")
            print(f"Stream URL: {self.stream_url}")
            
            # Create video capture object
            self.video_capture = cv2.VideoCapture(self.stream_url)
            
            if not self.video_capture.isOpened():
                print("Failed to open video capture, trying with CAP_FFMPEG backend...")
                self.video_capture = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)
            
            if self.video_capture.isOpened():
                # Test reading a frame
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    print(f"✅ Video stream started successfully!")
                    print(f"Frame size: {frame.shape}")
                    return True
                else:
                    print("❌ Could not read frame from stream")
                    return False
            else:
                print("❌ Failed to open video stream")
                return False
            
        except Exception as e:
            print(f"Failed to start video stream: {e}")
            return False
    
    def show_video(self):
        """Display video stream in a window with PTZ controls"""
        if not self.video_capture:
            print("Video stream not started")
            return
        
        print("\n🎥 Starting video display...")
        print("=" * 50)
        print("Controls:")
        print("  W/S: Tilt up/down")
        print("  A/D: Pan left/right") 
        print("  R: Reset to home position")
        print("  I: Show camera info")
        print("  Q: Quit")
        print("  SPACE: Take snapshot")
        print("=" * 50)
        
        frame_count = 0
        fps_start_time = time.time()
        fps_counter = 0
        
        while True:
            ret, frame = self.video_capture.read()
            
            if not ret:
                print(f"❌ Failed to read frame {frame_count}")
                frame_count += 1
                if frame_count > 5:
                    print("Too many failed frames, exiting...")
                    break
                time.sleep(0.1)
                continue
            
            frame_count = 0  # Reset counter on successful frame
            fps_counter += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - fps_start_time >= 1.0:
                fps = fps_counter / (current_time - fps_start_time)
                fps_start_time = current_time
                fps_counter = 0
            else:
                fps = 0
            
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
            
            # Add resolution info
            res_text = f"{width}x{height}"
            cv2.putText(frame, res_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, res_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            # Add control instructions
            controls_text = "WASD: Pan/Tilt | R: Home | SPACE: Snapshot | Q: Quit"
            text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.putText(frame, controls_text, (width - text_size[0] - 10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, controls_text, (width - text_size[0] - 10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Display the frame
            cv2.imshow('Tapo C200 Live Stream', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == ord('Q'):
                print("👋 Quitting...")
                break
            elif key == ord('w') or key == ord('W'):
                print("⬆️  Tilting up...")
                self.tilt_up()
            elif key == ord('s') or key == ord('S'):
                print("⬇️  Tilting down...")
                self.tilt_down()
            elif key == ord('a') or key == ord('A'):
                print("⬅️  Panning left...")
                self.pan_left()
            elif key == ord('d') or key == ord('D'):
                print("➡️  Panning right...")
                self.pan_right()
            elif key == ord('r') or key == ord('R'):
                print("🏠 Going to home position...")
                self.go_home()
            elif key == ord('i') or key == ord('I'):
                print("ℹ️  Getting camera info...")
                self.get_camera_info()
            elif key == ord(' '):  # Space bar
                print("📸 Taking snapshot...")
                self.save_snapshot()
        
        cv2.destroyAllWindows()
    
    def pan_left(self, speed=0.5):
        """Pan camera left"""
        if not self.ptz_service:
            print("PTZ service not available")
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = profile.token
            request.Velocity = {
                'PanTilt': {'x': -speed, 'y': 0.0},
                'Zoom': {'x': 0.0}
            }
            
            self.ptz_service.ContinuousMove(request)
            print("Panning left...")
            
            time.sleep(0.5)
            self.stop_ptz()
            
        except Exception as e:
            print(f"Failed to pan left: {e}")
    
    def pan_right(self, speed=0.5):
        """Pan camera right"""
        if not self.ptz_service:
            print("PTZ service not available")
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = profile.token
            request.Velocity = {
                'PanTilt': {'x': speed, 'y': 0.0},
                'Zoom': {'x': 0.0}
            }
            
            self.ptz_service.ContinuousMove(request)
            print("Panning right...")
            
            time.sleep(0.5)
            self.stop_ptz()
            
        except Exception as e:
            print(f"Failed to pan right: {e}")
    
    def tilt_up(self, speed=0.5):
        """Tilt camera up"""
        if not self.ptz_service:
            print("PTZ service not available")
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = profile.token
            request.Velocity = {
                'PanTilt': {'x': 0.0, 'y': speed},
                'Zoom': {'x': 0.0}
            }
            
            self.ptz_service.ContinuousMove(request)
            print("Tilting up...")
            
            time.sleep(0.5)
            self.stop_ptz()
            
        except Exception as e:
            print(f"Failed to tilt up: {e}")
    
    def tilt_down(self, speed=0.5):
        """Tilt camera down"""
        if not self.ptz_service:
            print("PTZ service not available")
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = profile.token
            request.Velocity = {
                'PanTilt': {'x': 0.0, 'y': -speed},
                'Zoom': {'x': 0.0}
            }
            
            self.ptz_service.ContinuousMove(request)
            print("Tilting down...")
            
            time.sleep(0.5)
            self.stop_ptz()
            
        except Exception as e:
            print(f"Failed to tilt down: {e}")
    
    def stop_ptz(self):
        """Stop PTZ movement"""
        if not self.ptz_service:
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = profile.token
            request.PanTilt = True
            request.Zoom = True
            
            self.ptz_service.Stop(request)
            
        except Exception as e:
            print(f"Failed to stop PTZ: {e}")
    
    def go_home(self):
        """Move camera to home position"""
        if not self.ptz_service:
            print("PTZ service not available")
            return
        
        try:
            profiles = self.media_service.GetProfiles()
            profile = profiles[0]
            
            request = self.ptz_service.create_type('GotoHomePosition')
            request.ProfileToken = profile.token
            
            self.ptz_service.GotoHomePosition(request)
            print("Moving to home position...")
            
        except Exception as e:
            print(f"Failed to go home: {e}")
    
    def get_camera_info(self):
        """Get comprehensive camera information"""
        try:
            device_service = self.camera.create_devicemgmt_service()
            device_info = device_service.GetDeviceInformation()
            
            print("\n=== Camera Information ===")
            print(f"Manufacturer: {device_info.Manufacturer}")
            print(f"Model: {device_info.Model}")
            print(f"Firmware Version: {device_info.FirmwareVersion}")
            print(f"Serial Number: {device_info.SerialNumber}")
            print(f"Hardware ID: {device_info.HardwareId}")
            print(f"Stream URL: {self.stream_url}")
            
            # Get network interfaces
            try:
                interfaces = device_service.GetNetworkInterfaces()
                print(f"\n=== Network Information ===")
                for interface in interfaces:
                    print(f"Interface: {interface.token}")
                    if hasattr(interface.Info, 'Name'):
                        print(f"  Name: {interface.Info.Name}")
                    if hasattr(interface, 'IPv4') and interface.IPv4:
                        for ipv4 in interface.IPv4.Config.Manual:
                            print(f"  IP: {ipv4.Address}")
            except Exception as e:
                print(f"Network info unavailable: {e}")
            
            # Get PTZ capabilities
            if self.ptz_service:
                try:
                    profiles = self.media_service.GetProfiles()
                    profile = profiles[0]
                    
                    print(f"\n=== PTZ Information ===")
                    print(f"PTZ Available: Yes")
                    print(f"Profile Token: {profile.token}")
                    
                    # Get PTZ configuration
                    ptz_config = self.ptz_service.GetConfiguration(profile.PTZConfiguration.token)
                    print(f"PTZ Node Token: {ptz_config.NodeToken}")
                    
                except Exception as e:
                    print(f"PTZ detailed info unavailable: {e}")
            else:
                print(f"\n=== PTZ Information ===")
                print("PTZ Available: No")
            
        except Exception as e:
            print(f"Failed to get camera info: {e}")
    
    def save_snapshot(self, filename=None):
        """Save a snapshot from the video stream"""
        if not self.video_capture:
            print("Video stream not available")
            return False
        
        ret, frame = self.video_capture.read()
        if ret:
            if not filename:
                filename = f"tapo_snapshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved as {filename}")
            return True
        else:
            print("Failed to capture snapshot")
            return False
    
    def disconnect(self):
        """Clean up and disconnect"""
        if self.video_capture:
            self.video_capture.release()
        cv2.destroyAllWindows()
        print("Disconnected from camera")

def main():
    # Camera configuration - Update these values
    CAMERA_IP = "192.168.1.143"  # Your camera's IP
    USERNAME = "admin123"        # Your camera username  
    PASSWORD = "admin123"        # Your camera password
    
    print("🎥 TP-Link Tapo C200 ONVIF Controller")
    print("=" * 50)
    print(f"Camera IP: {CAMERA_IP}")
    print(f"Username: {USERNAME}")
    print(f"Password: {'*' * len(PASSWORD)}")
    print()
    
    # Create controller instance
    controller = TapoC200Controller(CAMERA_IP, USERNAME, PASSWORD)
    
    try:
        # Connect to camera
        print("🔗 Connecting to camera...")
        if not controller.connect():
            print("\n❌ Failed to connect to camera. Please check:")
            print("1. Camera IP address is correct")
            print("2. Camera is powered on and connected to network")
            print("3. ONVIF is enabled in camera settings")
            print("4. Username and password are correct")
            return
        
        print("✅ Successfully connected!")
        
        # Get camera information
        controller.get_camera_info()
        
        # Start video stream
        print("\n📡 Starting video stream...")
        if not controller.start_video_stream():
            print("\n❌ Video stream failed, testing PTZ controls only...")
            
            # Test PTZ controls without video
            print("\nTesting PTZ controls (no video display):")
            print("Testing pan left...")
            controller.pan_left()
            time.sleep(1)
            
            print("Testing pan right...")
            controller.pan_right() 
            time.sleep(1)
            
            print("Testing tilt up...")
            controller.tilt_up()
            time.sleep(1)
            
            print("Testing tilt down...")
            controller.tilt_down()
            time.sleep(1)
            
            print("Returning to home position...")
            controller.go_home()
            
            return
        else:
            print("✅ Video stream started successfully!")
            
            # Test PTZ controls first
            print("\n🎮 Testing PTZ controls...")
            time.sleep(2)
            
            print("Testing pan left...")
            controller.pan_left(speed=0.3)
            time.sleep(2)
            
            print("Testing pan right...")
            controller.pan_right(speed=0.3)
            time.sleep(2)
            
            print("Testing tilt up...")
            controller.tilt_up(speed=0.3)
            time.sleep(2)
            
            print("Testing tilt down...")
            controller.tilt_down(speed=0.3)
            time.sleep(2)
            
            print("Returning to home position...")
            controller.go_home()
            time.sleep(3)
            
            print("✅ PTZ test completed!")
            
            # Show video with controls
            print("\n🎥 Starting video display with live controls...")
            controller.show_video()
        
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    
    finally:
        # Clean up
        print("🧹 Cleaning up...")
        controller.disconnect()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
