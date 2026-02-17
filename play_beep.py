
import os
import pyaudio
import wave


file_path = '/home/pi/person_alarm_ws/person_alarm_rtsp_ultrasonic/sounds/beep.wav'
    
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
    
    print(f"🔊 Playing beep")
    
    while data:
        stream.write(data)
        data = wf.readframes(chunk_size)
    
    # Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    
except FileNotFoundError:
    print(f"❌ Error: File '{file_path}' not found")
except Exception as e:
    print(f"❌ Error playing audio: {e}")