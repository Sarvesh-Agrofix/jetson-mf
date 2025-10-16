import os
import glob
import datetime
from typing import Optional
from fastapi import FastAPI, Response, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import asyncpg
import asyncio
from dotenv import load_dotenv

# ========================================
# CONFIGURATION
# ========================================

load_dotenv()  # Load DATABASE_URL from .env file
VIDEO_DIR = "./clips"
DATABASE_URL = os.getenv("DATABASE_URL")

# Ensure video directory exists
os.makedirs(VIDEO_DIR, exist_ok=True)

app = FastAPI(title="DMS Clip Server", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# DATABASE INITIALIZATION
# ========================================

async def init_db():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id SERIAL PRIMARY KEY,
            filename TEXT UNIQUE NOT NULL,
            filepath TEXT NOT NULL,
            size_bytes BIGINT NOT NULL,
            size_mb NUMERIC(10,2),
            modified TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    await conn.close()

@app.on_event("startup")
async def startup_event():
    await init_db()

# ========================================
# HELPER FUNCTION: STORE CLIP METADATA
# ========================================

async def store_clip_metadata(filename, filepath, size_bytes, size_mb, modified):
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        await conn.execute("""
            INSERT INTO clips (filename, filepath, size_bytes, size_mb, modified)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (filename)
            DO UPDATE SET 
                filepath = EXCLUDED.filepath,
                size_bytes = EXCLUDED.size_bytes,
                size_mb = EXCLUDED.size_mb,
                modified = EXCLUDED.modified;
        """, filename, filepath, size_bytes, size_mb, modified)
        await conn.close()
    except Exception as e:
        print(f"[DB ERROR] Failed to store clip metadata for {filename}: {e}")

# ========================================
# API ROUTES
# ========================================

@app.get("/")
async def root():
    return {
        "message": "Clip Streaming Server running with DB integration",
        "clips_directory": os.path.abspath(VIDEO_DIR),
        "endpoints": {
            "list": "/clips",
            "stream": "/clips/{filename}"
        }
    }

@app.get("/clips")
async def list_clips():
    if not os.path.exists(VIDEO_DIR):
        return {"clips": [], "total": 0}

    clips = []
    mp4_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    mp4_files = [f for f in mp4_files if not f.endswith(("_temp.mp4", "_web.mp4"))]

    for file_path in sorted(mp4_files, reverse=True):
        filename = os.path.basename(file_path)

        try:
            size = os.path.getsize(file_path)
            if size < 1024:
                continue

            mod_time = os.path.getmtime(file_path)
            created = datetime.datetime.fromtimestamp(mod_time)

            clip_info = {
                "filename": filename,
                "size_mb": round(size / (1024 * 1024), 2),
                "size_bytes": size,
                "modified": created.isoformat(),
                "url": f"/clips/{filename}",
                "playable": True
            }

            clips.append(clip_info)

            # Store metadata asynchronously
            asyncio.create_task(store_clip_metadata(
                filename,
                os.path.abspath(file_path),
                size,
                round(size / (1024 * 1024), 2),
                created
            ))

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue

    return {
        "clips": clips,
        "total": len(clips),
        "directory": os.path.abspath(VIDEO_DIR)
    }

@app.get("/clips/{filename}")
async def stream_clip(filename: str, range: Optional[str] = Header(None)):
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = os.path.join(VIDEO_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise HTTPException(status_code=404, detail="File is empty")

    start = 0
    end = file_size - 1

    if range:
        try:
            range_str = range.replace("bytes=", "")
            range_parts = range_str.split("-")
            if range_parts[0]:
                start = int(range_parts[0])
            if len(range_parts) > 1 and range_parts[1]:
                end = int(range_parts[1])
            if start > end or start < 0:
                raise ValueError
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid range format")

    chunk_size = end - start + 1
    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size)

    headers = {
        "Content-Type": "video/mp4",
        "Content-Length": str(chunk_size),
        "Accept-Ranges": "bytes",
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Cache-Control": "public, max-age=86400",
        "Access-Control-Allow-Origin": "*",
    }

    return Response(
        content=data,
        status_code=206 if range else 200,
        headers=headers,
        media_type="video/mp4"
    )

@app.options("/clips/{filename}")
async def options_clip(filename: str):
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
    print("DMS Clip Server with Neon DB integration")
    print("=" * 60)
    print("Serving from:", os.path.abspath(VIDEO_DIR))
    print("[INFO] Server starting on http://0.0.0.0:8080")
    asyncio.run(init_db())
    uvicorn.run(app, host="0.0.0.0", port=8080)
