import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import datetime
import os
import time
from queue import Queue, Empty
import signal
import sys
import threading
from collections import deque
import asyncio
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import glob

from config import *
from hardware_threads import gps_thread, imu_thread, shared_event_queue, gps_data
from vision_threads import video_capture_thread, process_frames_thread
from utils import AudioAlerter, log_event, manage_storage, draw_text_on_frame, get_ear_points, eye_aspect_ratio, mouth_aspect_ratio

import hashlib
import hmac
from fastapi import Depends, HTTPException, status, Request, Header
from typing import Optional

SECRET_KEY = b"super_secret_salt_here_1234567890"
TOKEN_VALIDITY_SECONDS = 30  # same as Flask version

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

async def require_auth(request):
    """Dependency to protect FastAPI routes with token auth."""
    # Try query params first
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

    # Try Authorization header: Bearer <token>:<timestamp>
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




#NEW ADDITION

# FastAPI app setup
app = FastAPI(title="Driver Monitoring System API", version="1.0.0")

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- Global State & Cleanup ---
cleanup_requested = False
latest_frame = None
frame_lock = threading.Lock()

def signal_handler(sig, frame):
    global cleanup_requested
    print("\n[INFO] Interrupt signal received, starting cleanup...")
    cleanup_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- FastAPI Endpoints ---
# Signed URL endpoint
@app.get("/get_signed_url")
async def get_signed_url():
    ts = int(time.time())
    token = generate_token(ts)
    signed_url = "/video_feed?token={}&ts={}".format(token, ts)
    return {"url": signed_url}



@app.get("/")
async def root():
    """Root endpoint with system status."""
    return {
        "message": "Driver Monitoring System API",
        "status": "running",
        "endpoints": {
            "video_feed": "/video_feed",
            "clips": "/clips"
        }
    }

@app.get("/video_feed")
async def video_feed(_auth=Depends(require_auth)):
    """Live MJPEG video streaming endpoint."""
    def generate_frames():
        while not cleanup_requested:
            with frame_lock:
                if latest_frame is not None:
                    # Encode frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', latest_frame)
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
    """List all available video clips."""
    if not os.path.exists(VIDEO_DIR):
        return {"clips": []}
    
    # Get all MP4 files in the video directory
    mp4_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    clips = []
    
    for file_path in sorted(mp4_files, reverse=True):  # Most recent first
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        mod_time = os.path.getmtime(file_path)
        
        clips.append({
            "filename": filename,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "created": datetime.datetime.fromtimestamp(mod_time).isoformat(),
            "download_url": "/clips/{}".format(filename)
        })
    
    return {"clips": clips}


@app.get("/clips/{filename}")
async def stream_clip(filename: str, range: Optional[str] = Header(None)):
    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path) or not filename.endswith(".mp4"):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    start = 0
    end = file_size - 1

    if range:
        # Example header: "bytes=0-1023"
        bytes_range = range.replace("bytes=", "").split("-")
        if bytes_range[0]:
            start = int(bytes_range[0])
        if bytes_range[1]:
            end = int(bytes_range[1])

    chunk_size = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
    }

    return Response(data, status_code=206, headers=headers)


def run_fastapi_server():
    import asyncio
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        print("[INFO] Starting FastAPI server on http://0.0.0.0:8000")
        import uvicorn
        config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        loop.run_until_complete(server.serve())
    except Exception as e:
        print("[ERROR] FastAPI server failed to start:", e)

# END NEW ADDITION

def main():
    global cleanup_requested, latest_frame
    
    # Start FastAPI server in a separate thread
    fastapi_thread = threading.Thread(target=run_fastapi_server, daemon=True)
    fastapi_thread.start()
    
    # Initialize queues and threads
    stop_event = threading.Event()
    video_frame_queue = Queue(maxsize=1)
    processing_input_queue = Queue(maxsize=1)
    processing_output_queue = Queue(maxsize=1)
    
    # Initialize and start the AudioAlerter as a managed thread
    audio_alerter = AudioAlerter()
    audio_alerter.start()
    
    threads = [
        threading.Thread(target=video_capture_thread, args=(video_frame_queue, stop_event)),
        threading.Thread(target=process_frames_thread, args=(processing_input_queue, processing_output_queue, stop_event)),
        threading.Thread(target=gps_thread, args=(stop_event,)),
        threading.Thread(target=imu_thread, args=(stop_event,))
    ]
    for t in threads: 
        t.daemon = True
        t.start()
    
    print("[INFO] Threads started, warming up...")
    time.sleep(3.0)
    
    # Play welcome message
    audio_alerter.speak_message("Driver monitoring system activated.")
    
    # State variables
    eye_closed_counter = 0
    mouth_open_counter = 0
    distraction_counter = 0
    head_drop_counter = 0
    calibration_frames = 60
    calibration_counter = 0
    calibration_ratios = []
    NEUTRAL_PITCH_RATIO = 0
    last_logged_event_type = None
    recent_yawns = deque()
    frequent_yawn_alert_triggered = False
    results = None
    
    video_writer = None
    clip_start_time = time.time()
    
    print("[INFO] Starting main loop with dashcam recording...")
    try:
        while not cleanup_requested:
            try:
                frame = video_frame_queue.get(timeout=1.0)
            except Empty:
                print("[WARNING] Main loop timeout waiting for frame.")
                continue

            frame_height, frame_width, _ = frame.shape
            
            # --- Dashcam recording logic (Modified to use MP4) ---
            if video_writer is None or (time.time() - clip_start_time) >= CLIP_DURATION_SECONDS:
                if video_writer is not None:
                    video_writer.release()
                    print("[INFO] Clip saved.")
                    threading.Thread(target=manage_storage).start()
                timestamp_fn = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(VIDEO_DIR, "clip_{}.mp4".format(timestamp_fn))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(filename, fourcc, FRAME_RATE, (frame_width, frame_height))
                clip_start_time = time.time()
                print("[INFO] Started new clip: {}".format(filename))

            try: 
                processing_input_queue.put_nowait(frame.copy())
            except: 
                pass
            try:
                new_results = processing_output_queue.get_nowait()
                if new_results is not None: 
                    results = new_results
            except Empty: 
                pass

            status = "Monitoring..."
            
            # Handle sensor events
            sensor_event = None
            try:
                sensor_event = shared_event_queue.get_nowait()
                log_event(sensor_event)
                audio_alerter.alert(sensor_event)
            except Empty: 
                pass

            # Face detection and analysis
            if results and results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                all_landmarks_px = [(int(pt.x * frame_width), int(pt.y * frame_height)) for pt in face_landmarks]
                
                # Calibration phase
                if NEUTRAL_PITCH_RATIO == 0:
                    status = "Calibrating... {}/{}".format(calibration_counter, calibration_frames)
                    if calibration_counter < calibration_frames:
                        eye_mid_y = (all_landmarks_px[INTER_EYE_L_INDEX][1] + all_landmarks_px[INTER_EYE_R_INDEX][1]) / 2
                        chin_y = all_landmarks_px[CHIN_INDEX][1]
                        nose_y = all_landmarks_px[NOSE_TIP_INDEX][1]
                        face_vertical_dist = chin_y - eye_mid_y
                        if face_vertical_dist > 0:
                            calibration_ratios.append((nose_y - eye_mid_y) / face_vertical_dist)
                        calibration_counter += 1
                    else:
                        NEUTRAL_PITCH_RATIO = np.mean(calibration_ratios) if calibration_ratios else 0.2
                        print("[INFO] Calibration complete. Neutral Pitch Ratio: {:.2f}".format(NEUTRAL_PITCH_RATIO))
                else:
                    # Main monitoring logic
                    face_alert_type = None
                    
                    # Mouth analysis (yawning)
                    top_lip_pt = all_landmarks_px[MOUTH_TOP_LIP_POINT]
                    bottom_lip_pt = all_landmarks_px[MOUTH_BOTTOM_LIP_POINT]
                    left_corner_pt = all_landmarks_px[MOUTH_LEFT_CORNER_POINT]
                    right_corner_pt = all_landmarks_px[MOUTH_RIGHT_CORNER_POINT]
                    mar = mouth_aspect_ratio([top_lip_pt, bottom_lip_pt, left_corner_pt, right_corner_pt])
                    
                    # Eye analysis (drowsiness)
                    ear = (eye_aspect_ratio(get_ear_points([all_landmarks_px[i] for i in LEFT_EYE_POINTS])) + 
                           eye_aspect_ratio(get_ear_points([all_landmarks_px[i] for i in RIGHT_EYE_POINTS]))) / 2.0
                    
                    if mar > MOUTH_AR_THRESH: 
                        mouth_open_counter += 1
                    else:
                        if mouth_open_counter >= MOUTH_AR_CONSEC_FRAMES: 
                            log_event("Yawn")
                            recent_yawns.append(datetime.datetime.now())
                        mouth_open_counter = 0
                    
                    # Eye drowsiness detection (not during yawning)
                    if not (mouth_open_counter > 3):
                        if ear < EYE_AR_THRESH:
                            eye_closed_counter += 1
                            if eye_closed_counter >= EYE_AR_CONSEC_FRAMES_ASLEEP: 
                                status, face_alert_type = "SLEEPING ALERT!", "Sleeping"
                            elif eye_closed_counter >= EYE_AR_CONSEC_FRAMES_DROWSY: 
                                status, face_alert_type = "Drowsiness Alert", "Drowsy"
                        else: 
                            eye_closed_counter = 0
                    else: 
                        eye_closed_counter = 0
                    
                    # Head pose analysis
                    eye_mid_y = (all_landmarks_px[INTER_EYE_L_INDEX][1] + all_landmarks_px[INTER_EYE_R_INDEX][1]) / 2
                    chin_y = all_landmarks_px[CHIN_INDEX][1]
                    nose_y = all_landmarks_px[NOSE_TIP_INDEX][1]
                    face_vertical_dist = chin_y - eye_mid_y
                    
                    if face_vertical_dist > 0:
                        current_pitch_ratio = (nose_y - eye_mid_y) / face_vertical_dist
                        if current_pitch_ratio > NEUTRAL_PITCH_RATIO + HEAD_PITCH_THRESH:
                            head_drop_counter += 1
                            if head_drop_counter >= HEAD_POSE_CONSEC_FRAMES: 
                                status, face_alert_type = "HEAD DROP ALERT!", "HeadDroop"
                        else: 
                            head_drop_counter = 0
                    
                    # Distraction detection (horizontal head movement)
                    nose_tip_px = all_landmarks_px[NOSE_TIP_INDEX]
                    neutral_nose_x_pos = frame_width / 2
                    if abs(nose_tip_px[0] - neutral_nose_x_pos) > (frame_width * DISTRACTION_THRESH_X_RATIO):
                        distraction_counter += 1
                        if distraction_counter >= HEAD_POSE_CONSEC_FRAMES: 
                            status, face_alert_type = "DISTRACTION ALERT!", "Distracted"
                    else: 
                        distraction_counter = 0
                    
                    # Frequent yawning detection
                    now = datetime.datetime.now()
                    while recent_yawns and (now - recent_yawns[0]).total_seconds() > YAWN_FREQ_WINDOW_SECONDS: 
                        recent_yawns.popleft()
                    if len(recent_yawns) >= YAWN_FREQ_COUNT and not frequent_yawn_alert_triggered:
                        status, face_alert_type, frequent_yawn_alert_triggered = "Frequent Yawning!", "FrequentYawn", True
                    elif len(recent_yawns) < YAWN_FREQ_COUNT: 
                        frequent_yawn_alert_triggered = False

                    # Handle alerts
                    if face_alert_type and face_alert_type != last_logged_event_type:
                        log_event(face_alert_type)
                        audio_alerter.alert(face_alert_type)
                        last_logged_event_type = face_alert_type
                    elif not face_alert_type:
                        last_logged_event_type = None
            else:
                status = "No Face Detected"
                NEUTRAL_PITCH_RATIO = 0
                calibration_counter = 0
                calibration_ratios.clear()
            
            # Display information on frame
            active_alert_type = last_logged_event_type or sensor_event
            alert_color = (0, 0, 255) if active_alert_type else (0, 255, 0)
            draw_text_on_frame(frame, "Status: {}".format(status), (10, 20), color=alert_color)
            
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            gps_str = "Lat: {} Lon: {}".format(gps_data['latitude'], gps_data['longitude'])
            speed_str = "Speed: {:.1f} km/h".format(gps_data['speed_kmh'])
            
            draw_text_on_frame(frame, timestamp_str, (10, frame_height - 10))
            draw_text_on_frame(frame, gps_str, (10, frame_height - 30))
            draw_text_on_frame(frame, speed_str, (frame_width - 150, 20))
            
            # Update latest frame for streaming (thread-safe)
            with frame_lock:
                latest_frame = frame.copy()
            
            if video_writer is not None:
                video_writer.write(frame)
            
            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cleanup_requested = True
    
    except (KeyboardInterrupt, SystemExit):
        cleanup_requested = True
    finally:
        print("[INFO] Cleaning up...")
        # Signal the audio alerter thread to stop
        audio_alerter.stop()
        stop_event.set()
        if video_writer is not None: 
            video_writer.release()
        cv2.destroyAllWindows()
        # Join all threads, including the audio alerter
        threads.append(audio_alerter)
        for thread in threads:
            if thread.is_alive(): 
                thread.join(timeout=1)
        print("[INFO] Cleanup complete.")


if __name__ == "__main__":
    main()