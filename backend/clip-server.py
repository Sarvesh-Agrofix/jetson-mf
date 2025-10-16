import os
import glob
import datetime
from typing import Optional
from fastapi import FastAPI, Response, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware

VIDEO_DIR = "./clips"

# Ensure video directory exists
os.makedirs(VIDEO_DIR, exist_ok=True)

app = FastAPI(title="DMS Clip Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Clip Streaming Server running",
        "clips_directory": os.path.abspath(VIDEO_DIR),
        "endpoints": {
            "list": "/clips",
            "stream": "/clips/{filename}"
        }
    }

@app.get("/clips")
async def list_clips():
    """List all saved clips with proper metadata"""
    if not os.path.exists(VIDEO_DIR):
        return {"clips": [], "total": 0}
    
    clips = []
    
    # Use glob to find all MP4 files
    mp4_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    
    # Skip temporary files
    mp4_files = [f for f in mp4_files if not f.endswith(("_temp.mp4", "_web.mp4"))]
    
    for file_path in sorted(mp4_files, reverse=True):
        filename = os.path.basename(file_path)
        
        try:
            size = os.path.getsize(file_path)
            
            # Skip empty or very small files (likely corrupt)
            if size < 1024:
                print(f"[DEBUG] Skipping small file: {filename} ({size} bytes)")
                continue
            
            mod_time = os.path.getmtime(file_path)
            created = datetime.datetime.fromtimestamp(mod_time).isoformat()
            
            clips.append({
                "filename": filename,
                "size_mb": round(size / (1024 * 1024), 2),
                "size_bytes": size,
                "modified": created,
                "url": "/clips/{}".format(filename),
                "playable": True
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {str(e)}")
            continue
    
    return {
        "clips": clips,
        "total": len(clips),
        "directory": os.path.abspath(VIDEO_DIR)
    }

@app.get("/clips/{filename}")
async def stream_clip(filename: str, range: Optional[str] = Header(None)):
    """
    Stream video clip with proper range support for seeking.
    Critical headers for browser video player compatibility.
    """
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    # Only allow MP4 files
    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(VIDEO_DIR, filename)
    
    # Verify file exists
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get file size
    try:
        file_size = os.path.getsize(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to get file size: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to access file")
    
    # Reject empty files
    if file_size == 0:
        raise HTTPException(status_code=404, detail="File is empty")
    
    start = 0
    end = file_size - 1
    
    # Parse HTTP Range header for seeking
    if range:
        try:
            # Format: "bytes=0-1023" or "bytes=0-"
            range_str = range.replace("bytes=", "")
            range_parts = range_str.split("-")
            
            if range_parts[0]:
                start = int(range_parts[0])
            
            if len(range_parts) > 1 and range_parts[1]:
                end = int(range_parts[1])
            else:
                end = file_size - 1
            
            # Validate range
            if start < 0 or end < 0:
                raise ValueError("Invalid range")
            if start >= file_size:
                raise HTTPException(status_code=416, detail="Range start exceeds file size")
            if end >= file_size:
                end = file_size - 1
            if start > end:
                raise HTTPException(status_code=416, detail="Invalid range")
                
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid range format")
    
    chunk_size = end - start + 1
    
    # Read the requested chunk
    try:
        with open(file_path, "rb") as f:
            f.seek(start)
            data = f.read(chunk_size)
        
        if not data:
            raise HTTPException(status_code=500, detail="Failed to read file data")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Failed to read file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to read file")
    
    # Critical headers for browser video player
    headers = {
        "Content-Type": "video/mp4",
        "Content-Length": str(chunk_size),
        "Accept-Ranges": "bytes",
        "Content-Range": "bytes {}-{}/{}".format(start, end, file_size),
        "Cache-Control": "public, max-age=86400",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
        "Access-Control-Allow-Headers": "Range, Content-Type",
    }
    
    # Return 206 Partial Content if range requested, otherwise 200 OK
    status_code = 206 if range else 200
    
    return Response(
        content=data,
        status_code=status_code,
        headers=headers,
        media_type="video/mp4"
    )

@app.options("/clips/{filename}")
async def options_clip(filename: str):
    """Handle CORS preflight requests"""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
            "Access-Control-Allow-Headers": "Range, Content-Type",
            "Access-Control-Max-Age": "3600",
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("DMS Clip Server - Fixed Version")
    print("=" * 60)
    print("Serving from: {}".format(os.path.abspath(VIDEO_DIR)))
    print("=" * 60)
    print("[INFO] Server starting on http://0.0.0.0:8080")
    print("[INFO] Make sure your clips are H.264 encoded MP4s")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")