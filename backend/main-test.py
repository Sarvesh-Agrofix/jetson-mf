"""
Lightweight DMS Test Backend with Real Camera - FIXED VERSION
Properly handles video codec and streaming for browser playback
"""

import time
import os
import glob
import datetime
import hmac
import hashlib
import threading
import cv2
import numpy as np
from typing import Optional
from fastapi import FastAPI, Response, Depends, HTTPException, status, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import subprocess
from pathlib import Path

# ============= CONFIGURATION =============
VIDEO_DIR = "./clips"
FRAME_RATE = 20
CLIP_DURATION_SECONDS = 60
CAMERA_INDEX = 0
SECRET_KEY = b"super_secret_salt_here_1234567890"
TOKEN_VALIDITY_SECONDS = 30

# Create video directory
os.makedirs(VIDEO_DIR, exist_ok=True)

# ============= AUTHENTICATION =============
def generate_token(timestamp):
    """Generate an HMAC-SHA256 token using timestamp."""
    return hmac.new(SECRET_KEY, str(timestamp).encode(), hashlib.sha256).hexdigest()

def verify_token(token, timestamp):
    """Verify that token matches expected and timestamp is recent."""
    now = int(time.time())
    if abs(now - timestamp) > TOKEN_VALIDITY_SECONDS:
        return False
    expected = generate_token(timestamp)
    return hmac.compare_digest(token, expected)

async def require_auth(request: Request):
    """Dependency to protect FastAPI routes with token auth."""
    token = request.query_params.get("token")
    ts_str = request.query_params.get("ts")

    if token and ts_str:
        try:
            ts = int(ts_str)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid timestamp")
        if not verify_token(token, ts):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        return

    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token, ts_str = auth_header.replace("Bearer ", "").split(":")
            ts = int(ts_str)
        except Exception:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid auth format")
        if not verify_token(token, ts):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
        return

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth credentials")

# ============= FASTAPI APP =============
app = FastAPI(title="DMS Test Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= GLOBAL STATE =============
latest_frame = None
frame_lock = threading.Lock()
stop_recording = False
camera = None
completed_clips = set()  # Track clips that are ready for playback

# ============= HELPER FUNCTIONS =============
def check_ffmpeg_available():
    """Check if FFmpeg is installed and available."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

def convert_to_h264(input_file):
    """
    Convert video to browser-compatible H.264 format using FFmpeg.
    This is critical for web playback.
    """
    if not os.path.exists(input_file):
        print(f"[ERROR] Input file not found: {input_file}")
        return False
    
    # Check file size
    file_size = os.path.getsize(input_file)
    if file_size < 1024:  # Less than 1KB
        print(f"[WARNING] File too small, likely corrupt: {input_file}")
        return False
    
    output_file = input_file.replace(".mp4", "_web.mp4")
    
    try:
        print(f"[INFO] Converting {os.path.basename(input_file)} to H.264...")
        
        # FFmpeg command optimized for web playback
        cmd = [
            'ffmpeg',
            '-i', input_file,
            '-c:v', 'libx264',           # H.264 video codec
            '-profile:v', 'baseline',     # Baseline profile for max compatibility
            '-level', '3.0',              # Level 3.0 for web
            '-pix_fmt', 'yuv420p',        # Standard pixel format
            '-movflags', '+faststart',    # Enable progressive download
            '-preset', 'medium',          # Balance speed/quality
            '-crf', '23',                 # Good quality
            '-an',                        # No audio (remove if you add audio later)
            '-y',                         # Overwrite output
            output_file
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=120,  # 2 minute timeout
            text=True
        )
        
        if result.returncode != 0:
            print(f"[ERROR] FFmpeg failed: {result.stderr}")
            return False
        
        # Verify output file was created and has content
        if not os.path.exists(output_file) or os.path.getsize(output_file) < 1024:
            print(f"[ERROR] Output file invalid: {output_file}")
            return False
        
        # Replace original with converted version
        os.replace(output_file, input_file)
        print(f"[SUCCESS] Converted {os.path.basename(input_file)} ({os.path.getsize(input_file) / 1024 / 1024:.2f} MB)")
        
        # Mark as ready for playback
        completed_clips.add(os.path.basename(input_file))
        return True
        
    except subprocess.TimeoutExpired:
        print(f"[ERROR] FFmpeg conversion timeout for {input_file}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        return False
    except FileNotFoundError:
        print("[ERROR] FFmpeg not installed! Install with: sudo apt install ffmpeg")
        return False
    except Exception as e:
        print(f"[ERROR] Conversion failed: {str(e)}")
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
            except:
                pass
        return False

def add_overlay_to_frame(frame):
    """Add timestamp and status overlay to frame."""
    if frame is None:
        return None
    
    # Create a copy to avoid modifying the original
    frame = frame.copy()
    height, width = frame.shape[:2]
    
    # Add semi-transparent black background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 90), (0, 0, 0), -1)
    cv2.rectangle(overlay, (0, height - 50), (width, height), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Add timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add status
    cv2.putText(frame, "Status: Monitoring", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Add mock GPS data
    cv2.putText(frame, "Lat: 17.6599 Lon: 75.9064", (10, height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, f"Speed: {45 + (int(time.time()) % 20):.1f} km/h", (width - 200, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    return frame

# ============= VIDEO RECORDING THREAD =============
def recording_thread():
    """Background thread that continuously records from camera."""
    global latest_frame, stop_recording, camera
    
    # Check FFmpeg availability
    has_ffmpeg = check_ffmpeg_available()
    if not has_ffmpeg:
        print("[WARNING] FFmpeg not available! Videos may not play in browsers.")
        print("[INFO] Install FFmpeg: sudo apt install ffmpeg")
    
    # Initialize camera
    camera = cv2.VideoCapture(CAMERA_INDEX)
    
    if not camera.isOpened():
        print(f"[ERROR] Could not open camera {CAMERA_INDEX}")
        print("[INFO] Trying alternative camera indices...")
        for idx in range(5):
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                print(f"[SUCCESS] Opened camera at index {idx}")
                break
        else:
            print("[ERROR] No camera found. Exiting recording thread.")
            return
    
    # Set camera properties for better quality
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camera.set(cv2.CAP_PROP_FPS, FRAME_RATE)
    
    # Get actual frame size
    ret, test_frame = camera.read()
    if not ret:
        print("[ERROR] Could not read from camera")
        camera.release()
        return
    
    frame_height, frame_width = test_frame.shape[:2]
    print(f"[INFO] Camera initialized: {frame_width}x{frame_height} @ {FRAME_RATE}fps")
    
    video_writer = None
    clip_start_time = time.time()
    frame_count = 0
    current_clip_path = None
    
    print("[INFO] Recording thread started")
    
    while not stop_recording:
        ret, frame = camera.read()
        
        if not ret:
            print("[WARNING] Failed to read frame from camera")
            time.sleep(0.1)
            continue
        
        # Add overlay
        frame_with_overlay = add_overlay_to_frame(frame)
        frame_count += 1
        
        # Update latest frame for live streaming
        with frame_lock:
            latest_frame = frame_with_overlay.copy()
        
        # Handle clip recording
        elapsed = time.time() - clip_start_time
        
        if video_writer is None or elapsed >= CLIP_DURATION_SECONDS:
            # Close previous clip
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                print(f"[INFO] Clip completed: {frame_count} frames written")
                
                # Convert to web format in background (if FFmpeg available)
                if has_ffmpeg and current_clip_path:
                    saved_path = current_clip_path
                    threading.Thread(
                        target=convert_to_h264,
                        args=(saved_path,),
                        daemon=True
                    ).start()
                
                frame_count = 0
            
            # Start new clip
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"clip_{timestamp_str}.mp4"
            current_clip_path = os.path.join(VIDEO_DIR, filename)
            
            # Use MP4V codec (will be converted to H.264 later)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                current_clip_path,
                fourcc,
                FRAME_RATE,
                (frame_width, frame_height)
            )
            
            if not video_writer.isOpened():
                print(f"[ERROR] Could not create video writer for {filename}")
                video_writer = None
            else:
                print(f"[INFO] Started new clip: {filename}")
                clip_start_time = time.time()
        
        # Write frame to clip
        if video_writer is not None:
            video_writer.write(frame_with_overlay)
        
        # Control frame rate
        time.sleep(1.0 / FRAME_RATE)
    
    # Cleanup
    if video_writer is not None:
        video_writer.release()
    if camera is not None:
        camera.release()
    print("[INFO] Recording thread stopped")

# ============= FASTAPI ENDPOINTS =============
@app.on_event("startup")
async def startup_event():
    """Start the recording thread when the app starts."""
    global stop_recording
    stop_recording = False
    recording_thread_obj = threading.Thread(target=recording_thread, daemon=True)
    recording_thread_obj.start()
    print("[INFO] FastAPI server started on http://0.0.0.0:8000")
    print("[INFO] Initializing camera...")
    time.sleep(2)

@app.on_event("shutdown")
async def shutdown_event():
    """Stop recording when the app shuts down."""
    global stop_recording
    stop_recording = True
    if camera is not None:
        camera.release()
    print("[INFO] Server shutting down...")

@app.get("/")
async def root():
    """Root endpoint with system status."""
    camera_status = "active" if (camera is not None and camera.isOpened()) else "inactive"
    ffmpeg_status = "available" if check_ffmpeg_available() else "missing"
    
    return {
        "message": "DMS Test Backend API",
        "status": "running",
        "camera_status": camera_status,
        "ffmpeg_status": ffmpeg_status,
        "clips_ready": len(completed_clips),
        "mode": "test",
        "endpoints": {
            "get_signed_url": "/get_signed_url",
            "video_feed": "/video_feed (requires auth)",
            "clips": "/clips",
            "download_clip": "/clips/{filename}"
        }
    }

@app.get("/get_signed_url")
async def get_signed_url():
    """Generate a signed URL for video streaming."""
    ts = int(time.time())
    token = generate_token(ts)
    signed_url = f"/video_feed?token={token}&ts={ts}"
    return {"url": signed_url}

@app.get("/video_feed")
async def video_feed(_auth=Depends(require_auth)):
    """Live MJPEG video streaming endpoint."""
    def generate_frames():
        while not stop_recording:
            with frame_lock:
                if latest_frame is not None:
                    ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/clips")
async def list_clips():
    """List all available video clips with status."""
    if not os.path.exists(VIDEO_DIR):
        return []
    
    mp4_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    clips = []
    
    for file_path in sorted(mp4_files, reverse=True):
        filename = os.path.basename(file_path)
        
        # Skip temporary files
        if filename.endswith("_temp.mp4") or filename.endswith("_web.mp4"):
            continue
        
        if not os.path.exists(file_path):
            continue
            
        file_size = os.path.getsize(file_path)
        
        # Skip empty or very small files
        if file_size < 1024:
            continue
            
        mod_time = os.path.getmtime(file_path)
        is_ready = filename in completed_clips
        
        clips.append({
            "filename": filename,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "created": datetime.datetime.fromtimestamp(mod_time).isoformat(),
            "download_url": f"/clips/{filename}",
            "ready": is_ready,
            "status": "ready" if is_ready else "processing"
        })
    
    return clips

@app.get("/clips/{filename}")
async def stream_clip(filename: str, range: Optional[str] = Header(None)):
    """
    Stream video clip with proper range support for seeking.
    Critical for browser video player functionality.
    """
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(VIDEO_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_size = os.path.getsize(file_path)
    
    # Ensure file has content
    if file_size == 0:
        raise HTTPException(status_code=404, detail="File is empty")
    
    start = 0
    end = file_size - 1

    # Parse range header for video seeking
    if range:
        range_str = range.replace("bytes=", "")
        range_parts = range_str.split("-")
        
        if range_parts[0]:
            start = int(range_parts[0])
        if len(range_parts) > 1 and range_parts[1]:
            end = int(range_parts[1])
        
        # Validate range
        if start >= file_size or end >= file_size or start > end:
            raise HTTPException(status_code=416, detail="Range not satisfiable")

    chunk_size = end - start + 1
    
    # Read and return the requested chunk
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(chunk_size)
    except Exception as e:
        print(f"[ERROR] Failed to read file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file")

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
        "Cache-Control": "public, max-age=3600",
    }

    status_code = 206 if range else 200
    return Response(data, status_code=status_code, headers=headers)

# ============= MAIN =============
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("DMS Test Backend with Real Camera - FIXED")
    print("=" * 60)
    print(f"Camera Index: {CAMERA_INDEX}")
    print(f"Recording to: {os.path.abspath(VIDEO_DIR)}")
    print(f"Clip Duration: {CLIP_DURATION_SECONDS}s")
    print(f"FFmpeg Available: {check_ffmpeg_available()}")
    print("=" * 60)
    print("\nStarting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")