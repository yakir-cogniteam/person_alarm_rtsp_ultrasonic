#!/usr/bin/env python3

import time
import paho.mqtt.client as mqtt
from pyrplidar import PyRPlidar
import json
import math
import threading


class LidarTest:
    def __init__(self, mqtt_broker="localhost", mqtt_port=1883, 
                 mqtt_topic="lidar/scan", lidar_port="/dev/ttyUSB0"):
        """
        Initialize LiDAR test scanner
        
        Args:
            mqtt_broker: MQTT broker address
            mqtt_port: MQTT broker port
            mqtt_topic: Topic to publish scan data
            lidar_port: Serial port for LiDAR (usually /dev/ttyUSB0)
        """
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic
        self.lidar_port = lidar_port
        
        # MQTT client
        self.mqtt_client = None
        self.mqtt_connected = False
        
        # LiDAR
        self.lidar = None
        self.running = False
        
        # Statistics
        self.scan_count = 0
        self.last_stats_time = time.time()
        self.scans_per_second = 0
        
    def _setup_mqtt(self):
        """Setup MQTT client"""
        try:
            self.mqtt_client = mqtt.Client(client_id="lidar_test")
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect
            
            print(f"🔗 Connecting to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}...")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            
        except Exception as e:
            print(f"❌ Failed to setup MQTT: {e}")
            self.mqtt_connected = False
            return False
        
        return True
    
    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            print("✅ Connected to MQTT broker")
            self.mqtt_connected = True
        else:
            print(f"❌ Failed to connect to MQTT broker. Code: {rc}")
            self.mqtt_connected = False
    
    def _on_mqtt_disconnect(self, client, userdata, rc):
        """Callback for MQTT disconnection"""
        print("⚠️  Disconnected from MQTT broker")
        self.mqtt_connected = False
    
    def _setup_lidar(self):
        """Initialize LiDAR connection"""
        try:
            print(f"🔗 Connecting to LiDAR on {self.lidar_port}...")
            
            self.lidar = PyRPlidar()
            self.lidar.connect(port=self.lidar_port, baudrate=115200, timeout=3)
            
            # Get device information
            info = self.lidar.get_info()
            print(f"📡 LiDAR Info: {info}")
            
            health = self.lidar.get_health()
            print(f"💚 LiDAR Health: {health}")
            
            samplerate = self.lidar.get_samplerate()
            print(f"📊 Sample Rate: {samplerate}")
            
            # Start motor
            print("🔄 Starting LiDAR motor...")
            self.lidar.set_motor_pwm(500)
            
            # Get scan modes
            scan_modes = self.lidar.get_scan_modes()
            print("\n📋 Available scan modes:")
            for idx, mode in enumerate(scan_modes):
                print(f"  Mode {idx}: {mode}")
            
            # Wait for motor to spin up
            print("⏳ Waiting for motor to stabilize...")
            time.sleep(2)
            
            print("✅ LiDAR initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize LiDAR: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _publish_scan(self, scan_points, scan_time):
        """
        Publish scan data via MQTT
        
        Args:
            scan_points: List of (angle, distance) tuples
            scan_time: Timestamp of the scan
        """
        if not self.mqtt_connected:
            return
        
        try:
            # Convert scan points to list of dicts with Cartesian coordinates
            points = []
            for angle, distance in scan_points:
                # Convert polar to Cartesian
                angle_rad = math.radians(angle)
                x = float((distance / 1000.0) * math.cos(angle_rad))  # Convert mm to meters
                y = float((distance / 1000.0) * math.sin(angle_rad))  # Convert mm to meters
                
                points.append({
                    'angle': float(angle),
                    'distance': float(distance),
                    'x': x,
                    'y': y
                })
            
            # Create scan message
            scan_msg = {
                'timestamp': scan_time,
                'point_count': len(points),
                'points': points,
                'scan_number': self.scan_count
            }
            
            # Convert to JSON and publish
            scan_json = json.dumps(scan_msg)
            self.mqtt_client.publish(self.mqtt_topic, scan_json)
            
        except Exception as e:
            print(f"❌ Error publishing scan: {e}")
    
    def _print_statistics(self):
        """Print scan statistics"""
        current_time = time.time()
        elapsed = current_time - self.last_stats_time
        
        if elapsed >= 1.0:  # Print stats every second
            self.scans_per_second = self.scan_count / elapsed
            print(f"📊 Stats: {self.scans_per_second:.2f} scans/sec | Total scans: {self.scan_count}")
            
            # Reset counters
            self.last_stats_time = current_time
            # Don't reset scan_count to keep total
    
    def run(self):
        """Main run loop - scan and publish LiDAR data"""
        
        # Setup MQTT
        if not self._setup_mqtt():
            return
        
        # Setup LiDAR
        if not self._setup_lidar():
            return
        
        # Use scan mode 2 (Boost) for better performance
        scan_mode = 2
        print(f"\n🚀 Starting LiDAR scanning with mode {scan_mode}...")
        print("Press Ctrl+C to stop\n")
        
        scan_generator = self.lidar.start_scan_express(scan_mode)()
        
        self.running = True
        scan_start_time = time.time()
        
        try:
            while self.running:
                # Collect points for one complete scan (360 degrees)
                scan_points = []
                scan_started = False
                
                for measurement in scan_generator:
                    # Parse measurement data
                    meas_str = str(measurement)
                    
                    try:
                        # Extract values
                        start_flag = 'True' in meas_str.split("'start_flag': ")[1].split(',')[0]
                        quality = int(meas_str.split("'quality': ")[1].split(',')[0])
                        angle = float(meas_str.split("'angle': ")[1].split(',')[0])
                        distance = float(meas_str.split("'distance': ")[1].split('}')[0])
                    except:
                        continue  # Skip malformed data
                    
                    # If we see a start flag and we've already started, we have a complete scan
                    if start_flag and scan_started:
                        break
                    
                    if start_flag:
                        scan_started = True
                        scan_start_time = time.time()
                    
                    # Collect valid points with quality filtering
                    MIN_QUALITY = 10  # Filter low quality points
                    if distance > 0 and quality > MIN_QUALITY:
                        scan_points.append((angle, distance))
                
                # Only publish if we have enough points (complete scan)
                if len(scan_points) > 50:
                    self.scan_count += 1
                    self._publish_scan(scan_points, scan_start_time)
                    self._print_statistics()
                else:
                    print(f"⚠️  Incomplete scan: {len(scan_points)} points - skipping")
                
                # Small delay to prevent CPU overload
                time.sleep(0.02)  # 20ms delay -> max ~50 scans/sec
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping...")
        except Exception as e:
            print(f"\n❌ Error occurred: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        self.running = False
        
        # Stop LiDAR
        if self.lidar:
            print("🛑 Stopping LiDAR motor...")
            try:
                self.lidar.set_motor_pwm(0)
                self.lidar.stop()
                self.lidar.disconnect()
                print("✅ LiDAR disconnected")
            except:
                pass
        
        # Stop MQTT
        if self.mqtt_client:
            print("🔌 Disconnecting MQTT...")
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("✅ MQTT disconnected")
        
        print("✨ Cleanup complete")


def main():
    # Configuration
    MQTT_BROKER = "localhost"  # Change to your MQTT broker IP if running remotely
    MQTT_PORT = 1883
    MQTT_TOPIC = "lidar/scan"
    LIDAR_PORT = "/dev/ttyUSB0"  # Change if your LiDAR is on a different port
    
    # Create LiDAR test instance
    lidar_test = LidarTest(
        mqtt_broker=MQTT_BROKER,
        mqtt_port=MQTT_PORT,
        mqtt_topic=MQTT_TOPIC,
        lidar_port=LIDAR_PORT
    )
    
    # Run
    lidar_test.run()


if __name__ == "__main__":
    main()